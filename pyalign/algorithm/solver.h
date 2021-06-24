#ifndef __PYALIGN_SOLVER__
#define __PYALIGN_SOLVER__

// Solver always computes one best alignment, but there might be multiple
// such alignments.

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xsort.hpp>

namespace pyalign {

struct ComputationGoal {
	struct score { // compute only score
	};

	struct alignment { // compute full traceback
	};
};

struct Direction {
	struct maximize {
		template<typename Value>
		static inline bool is_improvement(Value a, Value b) {
			return a > b;
		}

		static constexpr bool is_minimize() {
			return false;
		}
	};

	struct minimize {
		template<typename Value>
		static inline bool is_improvement(Value a, Value b) {
			return a < b;
		}

		static constexpr bool is_minimize() {
			return true;
		}
	};
};

class exceeded_length : public std::exception {
	const size_t m_len;
	const size_t m_max;

public:
	exceeded_length(const size_t p_len, const size_t p_max) :
		m_len(p_len), m_max(p_max) {
	}

	const size_t len() const {
		return m_len;
	}

	const size_t max() const {
		return m_max;
	}
};

class exceeded_implementation_length : public exceeded_length {
	const std::string m_err;

	static std::string to_text(const size_t p_len, const size_t p_max) {
		std::stringstream err;
		err << "requested maximum length " << p_len <<
			" exceeds maximum supported sequence length in this implementation " << p_max;
		return err.str();
	}

public:
	exceeded_implementation_length(const size_t p_len, const size_t p_max) :
		exceeded_length(p_len, p_max), m_err(to_text(p_len, p_max)) {
	}

	virtual char const *what() const noexcept {
		return m_err.c_str();
	}
};

class exceeded_configured_length : public exceeded_length {
	const std::string m_err;

	static std::string to_text(const size_t p_len, const size_t p_max) {
		std::stringstream err;
		err << "sequence of length " << p_len <<
			" exceeds configured maximum length " << p_max;
		return err.str();
	}

public:
	exceeded_configured_length(const size_t p_len, const size_t p_max) :
		exceeded_length(p_len, p_max), m_err(to_text(p_len, p_max)) {
	}

	virtual char const *what() const noexcept {
		return m_err.c_str();
	}
};

class Complexity {
	const std::string m_runtime;
	const std::string m_memory;

public:
	Complexity(const char *p_runtime, const char *p_memory) :
		m_runtime(p_runtime), m_memory(p_memory) {
	}

	const std::string &runtime() const {
		return m_runtime;
	}

	const std::string &memory() const {
		return m_memory;
	}
};

typedef std::shared_ptr<Complexity> ComplexityRef;

template<typename Value>
using GapTensorFactory = std::function<xt::xtensor<Value, 1>(size_t)>;

template<typename Index, typename Value, int LayerCount, int Layer>
class Matrix;

template<typename Index, typename Value, int LayerCount>
class MatrixFactory {
protected:
	friend class Matrix<Index, Value, LayerCount, 0>;
	friend class Matrix<Index, Value, LayerCount, 1>;
	friend class Matrix<Index, Value, LayerCount, 2>;

	struct Data {
		xt::xtensor<Value, 3> values;
		xt::xtensor<Index, 4> traceback;
		xt::xtensor<Index, 1> best_column;
	};

	const std::unique_ptr<Data> m_data;
	const size_t m_max_len_s;
	const size_t m_max_len_t;

	inline void check_size_against_max(const size_t p_len, const size_t p_max) const {
		if (p_len > p_max) {
			throw exceeded_configured_length(p_len, p_max);
		}
	}

	inline void check_size_against_implementation_limit(const size_t p_len) const {
		const size_t max = size_t(std::numeric_limits<Index>::max()) >> 1;
		if (p_len > max) {
			throw exceeded_implementation_length(p_len, max);
		}
	}

public:
	inline MatrixFactory(
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		m_data(std::make_unique<Data>()),
		m_max_len_s(p_max_len_s),
		m_max_len_t(p_max_len_t) {

		check_size_against_implementation_limit(p_max_len_s);
		check_size_against_implementation_limit(p_max_len_t);

		m_data->values.resize({
			LayerCount,
			m_max_len_s + 1,
			m_max_len_t + 1
		});
		m_data->traceback.resize({
			LayerCount,
			m_max_len_s + 1,
			m_max_len_t + 1,
			2
		});
		m_data->best_column.resize({
			m_max_len_s + 1
		});
	}

	template<int Layer>
	inline Matrix<Index, Value, LayerCount, Layer> make(
		const Index len_s, const Index len_t) const;

	inline Index max_len_s() const {
		return m_max_len_s;
	}

	inline Index max_len_t() const {
		return m_max_len_t;
	}

	template<int Layer>
	inline auto values() const {
		return xt::view(m_data->values, Layer, xt::all(), xt::all());
	}

	inline auto values_all(const Index len_s, const Index len_t) const {
		return xt::view(
			m_data->values,
			xt::all(), xt::range(0, len_s + 1), xt::range(0, len_t + 1));
	}

	inline auto traceback_all(const Index len_s, const Index len_t) const {
		return xt::view(
			m_data->traceback,
			xt::all(), xt::range(0, len_s + 1), xt::range(0, len_t + 1), xt::all());
	}
};

template<int Layer, int i0, int j0, typename Index, typename Tensor>
inline auto shifted_indices(Tensor &v) {
	// a custom view based on xt::xtensor to make sure that negative indexes, e.g.
	// m(-1, 2), are handled correctly. this is not guaranteed in xtensor.

	return [&v] (const Index i, const Index j) -> typename Tensor::reference {
		return v(Layer, i + i0, j + j0);
	};
}

template<typename Index, typename Value, int LayerCount, int Layer>
class Matrix {
	const MatrixFactory<Index, Value, LayerCount> &m_factory;
	const Index m_len_s;
	const Index m_len_t;

public:
	inline Matrix(
		const MatrixFactory<Index, Value, LayerCount> &factory,
		const Index len_s,
		const Index len_t) :

	    m_factory(factory),
	    m_len_s(len_s),
	    m_len_t(len_t) {
	}

	inline Index len_s() const {
		return m_len_s;
	}

	inline Index len_t() const {
		return m_len_t;
	}

	template<int i0, int j0>
	inline auto values_n() const {
		return shifted_indices<Layer, i0, j0, Index>(
			m_factory.m_data->values);
	}

	template<int i0, int j0>
	inline auto traceback_n() const {
		return shifted_indices<Layer, i0, j0, Index>(
			m_factory.m_data->traceback);
	}

	template<int i0, int j0>
	inline auto values() const {
		return xt::view(
			m_factory.m_data->values,
			Layer,
			xt::range(i0, m_len_s + 1),
			xt::range(j0, m_len_t + 1));
	}

	template<int i0, int j0>
	inline auto traceback() const {
		return xt::view(
			m_factory.m_data->traceback,
			Layer,
			xt::range(i0, m_len_s + 1),
			xt::range(j0, m_len_t + 1));
	}

	inline auto best_column() const {
		return xt::view(
			m_factory.m_data->best_column,
			xt::range(0, m_len_s));
	}
};

template<typename Index, typename Value, int LayerCount>
template<int Layer>
inline Matrix<Index, Value, LayerCount, Layer> MatrixFactory<Index, Value, LayerCount>::make(
	const Index len_s, const Index len_t) const {

	check_size_against_max(len_s, m_max_len_s);
	check_size_against_max(len_t, m_max_len_t);
	return Matrix<Index, Value, LayerCount, Layer>(*this, len_s, len_t);
}

template<typename V>
inline size_t argmax(const V &v) {
	// we do not use xt::argmax here,
	// since we want a guaranteed behaviour
	// (lowest index) on ties.

	const size_t n = v.shape(0);

	auto best = v(0);
	size_t best_i = 0;

	for (size_t i = 1; i < n; i++) {
		const auto x = v(i);
		if (x > best) {
			best = x;
			best_i = i;
		}
	}

	return best_i;
}


template<typename Alignment>
class build_alignment {
	Alignment &m_alignment;

public:
	inline build_alignment(Alignment &p_alignment) :
		m_alignment(p_alignment) {
	}

	template<typename Index>
	inline void begin(
		const Index len_s,
		const Index len_t) {

		m_alignment.resize(len_s, len_t);
	}

	template<typename Index>
	inline void step(
		const Index last_u,
		const Index last_v,
		const Index u,
		const Index v) {

		if (u != last_u && v != last_v) {
			m_alignment.add_edge(last_u, last_v);
		}
	}
};

template<typename Index>
class build_path {
	typedef xt::xtensor_fixed<Index, xt::xshape<2>> Coord;

	std::vector<Coord> m_path;

public:
	inline void begin(
		const Index len_s,
		const Index len_t) {

		m_path.reserve(len_s + len_t);
	}

	inline void step(
		const Index last_u,
		const Index last_v,
		const Index u,
		const Index v) {

		if (m_path.empty()) {
			m_path.push_back(Coord{last_u, last_v});
			m_path.push_back(Coord{u, v});
		} else {
			if (m_path.back()(0) != last_u) {
				throw std::runtime_error(
					"internal error in traceback generation");
			}
			if (m_path.back()(1) != last_v) {
				throw std::runtime_error(
					"internal error in traceback generation");
			}
			m_path.push_back(Coord{u, v});
		}
	}

	inline xt::xtensor<Index, 2> path() const {
		xt::xtensor<Index, 2> path;
		path.resize({m_path.size(), 2});
		for (size_t i = 0; i < m_path.size(); i++) {
			xt::view(path, i, xt::all()) = m_path[i];
		}
		return path;
	}
};

template<typename... BuildRest>
class build_multiple {
public:
	inline build_multiple(const BuildRest&... args) {
	}

	template<typename Index>
	inline void begin(
		const Index len_s,
		const Index len_t) {
	}

	template<typename Index>
	inline void step(
		const Index last_u,
		const Index last_v,
		const Index u,
		const Index v) {
	}

	template<int i>
	const auto &get() const {
		throw std::invalid_argument(
			"illegal index in build_multiple");
	}
};

template<typename BuildHead, typename... BuildRest>
class build_multiple<BuildHead, BuildRest...> {
	BuildHead m_head;
	build_multiple<BuildRest...> m_rest;

public:
	inline build_multiple(const BuildHead &arg1, const BuildRest&... args) :
		m_head(arg1), m_rest(args...) {
	}

	template<typename Index>
	inline void begin(
		const Index len_s,
		const Index len_t) {

		m_head.begin(len_s, len_t);
		m_rest.begin(len_s, len_t);
	}

	template<typename Index>
	inline void step(
		const Index last_u,
		const Index last_v,
		const Index u,
		const Index v) {

		m_head.step(last_u, last_v, u, v);
		m_rest.step(last_u, last_v, u, v);
	}

	template<int i>
	const auto &get() const {
		return m_rest.template get<i-1>();
	}

	template<>
	const auto &get<0>() const {
		return m_head;
	}
};

class build_nothing {
public:
	typedef ssize_t Index;

	inline void begin(
		const Index len_s,
		const Index len_t) {
	}

	inline void step(
		const Index last_u,
		const Index last_v,
		const Index u,
		const Index v) {
	}
};


template<typename Direction, typename Index, typename Value>
class Accumulator {
private:
	Value m_score;

public:
	inline void set(
		const Value score,
		const Index u,
		const Index v) {

		m_score = score;
	}

	inline void push(
		const Value score,
		const Index u,
		const Index v) {

		if (Direction::is_improvement(score, m_score)) {
			m_score = score;
		}
	}

	inline void add(const float score) {
		m_score += score;
	}

	template<typename ScoreRef, typename TracebackRef>
	inline void write(ScoreRef &&r_score, TracebackRef &&r_traceback) const {
		r_score = m_score;
	}
};

template<typename Direction, typename Index, typename Value>
class TracingAccumulator {
	typedef xt::xtensor_fixed<Index, xt::xshape<2>> Coord;

	Value m_score;
	Coord m_traceback;

public:
	inline void set(
		const Value score,
		const Index u,
		const Index v) {

		m_score = score;
		m_traceback[0] = u;
		m_traceback[1] = v;
	}

	inline void push(
		const Value score,
		const Index u,
		const Index v) {

		if (Direction::is_improvement(score, m_score)) {
			m_score = score;
			m_traceback[0] = u;
			m_traceback[1] = v;
		}
	}

	inline void add(const float score) {
		m_score += score;
	}

	template<typename ScoreRef, typename TracebackRef>
	inline void write(ScoreRef &&r_score, TracebackRef &&r_traceback) const {
		r_score = m_score;
		r_traceback = m_traceback;
	}
};


template<typename Index, typename Value>
class Local {
private:
	const Value m_zero;

	template<typename Path>
	class Backtracer {
	public:
		template<typename Matrix>
		inline static std::pair<Index, Index> build(
			Path &path,
			const Matrix &matrix,
			const float zero,
			const Index u0, const Index v0) {

			const auto len_s = matrix.len_s();
			const auto len_t = matrix.len_t();
			path.begin(len_s, len_t);

			const auto values = matrix.template values_n<1, 1>();
			const auto traceback = matrix.template traceback<1, 1>();

			Index u = u0;
			Index v = v0;

			while (u >= 0 && v >= 0 && values(u, v) > zero) {
				const Index last_u = u;
				const Index last_v = v;

				const auto t = xt::view(traceback, u, v, xt::all());
				u = t(0);
				v = t(1);

				path.step(last_u, last_v, u, v);
			}

			return std::make_pair(u, v);
		}
	};

public:
	template<typename Direction, typename Goal>
	struct AccumulatorFactory {
		typedef TracingAccumulator<Direction, Index, Value> Accumulator;
	};

	typedef Value ValueType;

	inline Local(const Value p_zero) : m_zero(p_zero) {
	}

	inline const char *name() const {
		return "local";
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		p_vector.fill(0);
	}

	template<typename Accumulator>
	inline void update_acc(Accumulator &acc) const {
		acc.push(m_zero, -1, -1);
	}

	template<typename Matrix, typename Path>
	inline Value traceback(
		Matrix &matrix,
		Path &path) const {

		const auto len_s = matrix.len_s();
		//const auto len_t = matrix.len_t();

		const auto values = matrix.template values_n<1, 1>();
		auto best_column = matrix.best_column();

		const auto zero_similarity = m_zero;

		best_column = xt::argmax(matrix.template values<1, 1>(), 1);

		Value score = zero_similarity;
		Index best_u = 0, best_v = 0;

		for (Index u = 0; u < len_s; u++) {
			const Index v = best_column(u);
			const Value s = values(u, v);
			if (s > score) {
				score = s;
				best_u = u;
				best_v = v;
			}
		}

		if (score <= zero_similarity) {
			return 0;
		}

		Index u, v;
		std::tie(u, v) = Backtracer<Path>::build(
			path, matrix, m_zero, best_u, best_v);

		if (u >= 0 && v >= 0) {
			return score - values(u, v);
		} else {
			return score;
		}
	}
};


template<typename Index, typename Value>
class Global {
	template<typename Path>
	class Backtracer {
	public:
		template<typename Matrix>
		inline static void build(
			Path &path,
			const Matrix &matrix,
			const Index u0, const Index v0) {

			const auto len_s = matrix.len_s();
			const auto len_t = matrix.len_t();
			path.begin(len_s, len_t);

			const auto traceback = matrix.template traceback<1, 1>();

			Index u = u0;
			Index v = v0;

			while (u >= 0 && v >= 0) {
				const Index last_u = u;
				const Index last_v = v;

				const auto t = xt::view(traceback, u, v, xt::all());
				u = t(0);
				v = t(1);

				path.step(last_u, last_v, u, v);
			}
		}
	};

	template<>
	class Backtracer<build_nothing> {
	public:
		template<typename Matrix>
		inline static void build(
			build_nothing &path,
			const Matrix &matrix,
			const Index u0, const Index v0) {
		}
	};

public:
	template<typename Direction, typename Goal>
	struct AccumulatorFactory {
	};

	template<typename Direction>
	struct AccumulatorFactory<Direction, ComputationGoal::score> {
		typedef Accumulator<Direction, Index, Value> Accumulator;
	};

	template<typename Direction>
	struct AccumulatorFactory<Direction, ComputationGoal::alignment> {
		typedef TracingAccumulator<Direction, Index, Value> Accumulator;
	};

	typedef Value ValueType;

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		p_vector = xt::view(
			p_gap_cost, xt::range(0, p_vector.size()));
	}

	template<typename Accumulator>
	inline void update_acc(Accumulator &) const {
	}

	template<typename Matrix, typename Path>
	inline Value traceback(
		Matrix &matrix,
		Path &path) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.template values_n<1, 1>();

		const Index u = len_s - 1;
		const Index v = len_t - 1;

		Backtracer<Path>::build(path, matrix, u, v);

		return values(u, v);
	}
};


template<typename Index, typename Value>
class Semiglobal {
	template<typename Path>
	class Backtracer {
	public:
		template<typename Matrix>
		inline static void build(
			Path &path,
			const Matrix &matrix,
			const Index u0, const Index v0) {

			const auto len_s = matrix.len_s();
			const auto len_t = matrix.len_t();
			path.begin(len_s, len_t);

			const auto traceback = matrix.template traceback<1, 1>();

			Index u = u0;
			Index v = v0;

			while (u >= 0 && v >= 0) {
				const Index last_u = u;
				const Index last_v = v;

				const auto t = xt::view(traceback, u, v, xt::all());
				u = t(0);
				v = t(1);

				path.step(last_u, last_v, u, v);
			}
		}
	};

	template<>
	class Backtracer<build_nothing> {
	public:
		template<typename Matrix>
		inline static void build(
			build_nothing &path,
			const Matrix &matrix,
			const Index u0, const Index v0) {
		}
	};

public:
	template<typename Direction, typename Goal>
	struct AccumulatorFactory {
	};

	template<typename Direction>
	struct AccumulatorFactory<Direction, ComputationGoal::score> {
		typedef Accumulator<Direction, Index, Value> Accumulator;
	};

	template<typename Direction>
	struct AccumulatorFactory<Direction, ComputationGoal::alignment> {
		typedef TracingAccumulator<Direction, Index, Value> Accumulator;
	};

	typedef Value ValueType;

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		p_vector.fill(0);
	}

	template<typename Accumulator>
	inline void update_acc(Accumulator &) const {
	}

	template<typename Matrix, typename Path>
	inline Value traceback(
		Matrix &matrix,
		Path &path) const {

		const auto len_s = matrix.len_s();
		const auto len_t = matrix.len_t();

		const auto values = matrix.template values_n<1, 1>();

		const Index last_row = len_s - 1;
		const Index last_col = len_t - 1;

		const auto values_non_neg_ij = matrix.template values<1, 1>();
		const Index best_col_in_last_row = argmax(xt::row(values_non_neg_ij, last_row));
		const Index best_row_in_last_col = argmax(xt::col(values_non_neg_ij, last_col));

		Index u;
		Index v;

		if (values(best_row_in_last_col, last_col) > values(last_row, best_col_in_last_row)) {
			u = best_row_in_last_col;
			v = last_col;
		} else {
			u = last_row;
			v = best_col_in_last_row;
		}

		Backtracer<Path>::build(path, matrix, u, v);

		return values(u, v);
	}
};

template<typename Index, typename Value>
class Solution {
public:
	xt::xtensor<Value, 3> m_values;
	xt::xtensor<Index, 4> m_traceback;
	xt::xtensor<Index, 2> m_path;
	Value m_score;
	ComplexityRef m_complexity;

	const auto &values() const {
		return m_values;
	}

	const auto &traceback() const {
		return m_traceback;
	}

	const auto &path() const {
		return m_path;
	}

	const auto score() const {
		return m_score;
	}

	const auto &complexity() const {
		return m_complexity;
	}
};

template<typename Index, typename Value>
using SolutionRef = std::shared_ptr<Solution<Index, Value>>;


template<typename Locality, typename Index, int LayerCount>
class Solver {
public:
	typedef typename Locality::ValueType Value;

protected:
	const Locality m_locality;
	MatrixFactory<Index, Value, LayerCount> m_factory;
	const ComplexityRef m_complexity;

public:
	inline Solver(
		const Locality &p_locality,
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const ComplexityRef &p_complexity) :

		m_locality(p_locality),
		m_factory(p_max_len_s, p_max_len_t),
		m_complexity(p_complexity) {
	}

	inline Index max_len_s() const {
		return m_factory.max_len_s();
	}

	inline Index max_len_t() const {
		return m_factory.max_len_t();
	}

	template<int Layer>
	inline auto matrix(const Index len_s, const Index len_t) {
		return m_factory.make<Layer>(len_s, len_t);
	}

	inline Value score(
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = m_factory.template make<0>(len_s, len_t);
		build_nothing nothing;
		return m_locality.traceback(matrix, nothing);
	}

	template<typename Alignment>
	inline Value alignment(
		const size_t len_s,
		const size_t len_t,
		Alignment &alignment) const {

		auto matrix = m_factory.template make<0>(len_s, len_t);
		build_alignment<Alignment> build(alignment);
		return m_locality.traceback(matrix, build);
	}

	template<typename Alignment>
	SolutionRef<Index, Value> solution(
		const size_t len_s,
		const size_t len_t,
		Alignment &alignment) const {

		const SolutionRef<Index, Value> solution =
			std::make_shared<Solution<Index, Value>>();

		solution->m_values = m_factory.values_all(len_s, len_t);
		solution->m_traceback = m_factory.traceback_all(len_s, len_t);

		auto build = build_multiple<build_path<Index>, build_alignment<Alignment>>(
			build_path<Index>(), build_alignment<Alignment>(alignment)
		);

		auto matrix = m_factory.template make<0>(len_s, len_t);
		const auto score = m_locality.traceback(matrix, build);
		solution->m_path = build.template get<0>().path();
		solution->m_score = score;

		solution->m_complexity = m_complexity;

		return solution;
	}
};

template<typename Locality, typename Index, int LayerCount>
using AlignmentSolver = Solver<Locality, Index, LayerCount>;

template<typename Direction, typename Locality, typename Index=int16_t>
class LinearGapCostSolver final : public AlignmentSolver<Locality, Index, 1> {
	// For global alignment, we pose the problem as a Needleman-Wunsch problem, but follow the
	// implementation of Sankoff and Kruskal.

	// For local alignments, we modify the problem for local  by adding a fourth zero case and
	// modifying the traceback (see Aluru or Hendrix).

	// Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable
	// to the search for similarities in the amino acid sequence of two proteins.
	// Journal of Molecular Biology, 48(3), 443–453. https://doi.org/10.1016/0022-2836(70)90057-4

	// Smith, T. F., & Waterman, M. S. (1981). Identification of common
	// molecular subsequences. Journal of Molecular Biology, 147(1), 195–197.
	// https://doi.org/10.1016/0022-2836(81)90087-5

	// Sankoff, D. (1972). Matching Sequences under Deletion/Insertion Constraints. Proceedings of
	// the National Academy of Sciences, 69(1), 4–6. https://doi.org/10.1073/pnas.69.1.4

	// Kruskal, J. B. (1983). An Overview of Sequence Comparison: Time Warps,
	// String Edits, and Macromolecules. SIAM Review, 25(2), 201–237. https://doi.org/10.1137/1025045

	// Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
	// Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275

	// Hendrix, D. A. Applied Bioinformatics. https://open.oregonstate.education/appliedbioinformatics/.

public:
	typedef typename AlignmentSolver<Locality, Index, 1>::Value Value;

private:
	const Value m_gap_cost_s;
	const Value m_gap_cost_t;

public:
	typedef Locality LocalityType;
	typedef Index IndexType;
	typedef Value GapCostSpec;

	inline LinearGapCostSolver(
		const Locality &p_locality,
		const Value p_gap_cost_s,
		const Value p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		AlignmentSolver<Locality, Index, 1>(
			p_locality,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<Complexity>("n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto values = this->m_factory.template values<0>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		p_locality.init_border_case(
			xt::view(values, xt::all(), 0),
			xt::arange<Index>(0, p_max_len_s + 1) * p_gap_cost_s * gap_sgn);
		p_locality.init_border_case(
			xt::view(values, 0, xt::all()),
			xt::arange<Index>(0, p_max_len_t + 1) * p_gap_cost_t * gap_sgn);
	}

	inline Value gap_cost_s(const size_t len) const {
		return m_gap_cost_s * len;
	}

	inline Value gap_cost_t(const size_t len) const {
		return m_gap_cost_t * len;
	}

	template<typename ComputationGoal, typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = this->m_factory.template make<0>(len_s, len_t);

		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

				typename Locality::template AccumulatorFactory<
					Direction, ComputationGoal>::Accumulator acc;

				acc.set(
					values(u - 1, v - 1) + pairwise(u, v),
					u - 1, v - 1);

				acc.push(
					values(u - 1, v) + this->m_gap_cost_s * gap_sgn,
					u - 1, v);

				acc.push(
					values(u, v - 1) + this->m_gap_cost_t * gap_sgn,
					u, v - 1);

				this->m_locality.update_acc(acc);

				acc.write(
					values(u, v),
					xt::view(traceback, u, v, xt::all()));
			}
		}
	}
};

template<typename Value>
inline void check_gap_tensor_shape(const xt::xtensor<Value, 1> &tensor, const size_t expected_len) {
	if (tensor.shape(0) != expected_len) {
		std::stringstream s;
		s << "expected gap cost tensor length of " << expected_len << ", got " << tensor.shape(0);
		throw std::invalid_argument(s.str());
	}
}

template<typename Direction, typename Locality, typename Index=int16_t>
class GeneralGapCostSolver final : public AlignmentSolver<Locality, Index, 1> {
	// Our implementation follows what is sometimes referred to as Waterman-Smith-Beyer, i.e.
	// an O(n^3) algorithm for generic gap costs. Waterman-Smith-Beyer generates a local alignment.

	// We use the same implementation approach as in LinearGapCostSolver to differentiate
	// between local and global alignments.

	// Waterman, M. S., Smith, T. F., & Beyer, W. A. (1976). Some biological sequence metrics.
	// Advances in Mathematics, 20(3), 367–387. https://doi.org/10.1016/0001-8708(76)90202-4

	// Aluru, S. (Ed.). (2005). Handbook of Computational Molecular Biology.
	// Chapman and Hall/CRC. https://doi.org/10.1201/9781420036275

	// Hendrix, D. A. Applied Bioinformatics. https://open.oregonstate.education/appliedbioinformatics/.

public:
	typedef typename AlignmentSolver<Locality, Index, 1>::Value Value;

private:
	const xt::xtensor<Value, 1> m_gap_cost_s;
	const xt::xtensor<Value, 1> m_gap_cost_t;

public:
	typedef Locality LocalityType;
	typedef Index IndexType;
	typedef GapTensorFactory<Value> GapCostSpec;

	inline GeneralGapCostSolver(
		const Locality &p_locality,
		const GapTensorFactory<Value> &p_gap_cost_s,
		const GapTensorFactory<Value> &p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		AlignmentSolver<Locality, Index, 1>(
			p_locality,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<Complexity>("n^3", "n^2")),
		m_gap_cost_s(p_gap_cost_s(p_max_len_s + 1)),
		m_gap_cost_t(p_gap_cost_t(p_max_len_t + 1)) {

		check_gap_tensor_shape(m_gap_cost_s, p_max_len_s + 1);
		check_gap_tensor_shape(m_gap_cost_t, p_max_len_t + 1);

		auto values = this->m_factory.template values<0>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		p_locality.init_border_case(
			xt::view(values, xt::all(), 0),
			m_gap_cost_s * gap_sgn);

		p_locality.init_border_case(
			xt::view(values, 0, xt::all()),
			m_gap_cost_t * gap_sgn);
	}

	inline Value gap_cost_s(const size_t len) const {
		PPK_ASSERT(len < m_gap_cost_s.shape(0));
		return m_gap_cost_s(len);
	}

	inline Value gap_cost_t(const size_t len) const {
		PPK_ASSERT(len < m_gap_cost_t.shape(0));
		return m_gap_cost_t(len);
	}

	template<typename ComputationGoal, typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = this->m_factory.template make<0>(len_s, len_t);

		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

				typename Locality::template AccumulatorFactory<Direction,
					ComputationGoal>::Accumulator acc;

				acc.set(
					values(u - 1, v - 1) + pairwise(u, v),
					u - 1, v - 1);

				for (Index k = -1; k < u; k++) {
					acc.push(
						values(k, v) + this->m_gap_cost_s(u - k) * gap_sgn,
						k, v);
				}

				for (Index k = -1; k < v; k++) {
					acc.push(
						values(u, k) + this->m_gap_cost_t(v - k) * gap_sgn,
						u, k);
				}

				this->m_locality.update_acc(acc);

				acc.write(
					values(u, v),
					xt::view(traceback, u, v, xt::all()));
			}
		}
	}
};

template<typename Direction, typename Index=int16_t, typename Value=float>
class DynamicTimeSolver final : public Solver<Global<Index, Value>, Index, 1> {
public:
	typedef Index IndexType;

	inline DynamicTimeSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		Solver<Global<Index, Value>, Index, 1>(
			Global<Index, Value>(),
			p_max_len_s,
			p_max_len_t,
			std::make_shared<Complexity>("n^2", "n^2")) {

		auto values = this->m_factory.template values<0>();
		values.fill(std::numeric_limits<Value>::infinity() * (Direction::is_minimize() ? 1 : -1));
		values.at(0, 0) = 0;
	}

	template<typename ComputationGoal, typename Pairwise>
	void solve(
		const Pairwise &pairwise, // similarity or distance, depends on Direction
		const size_t len_s,
		const size_t len_t) const {

		// Müller, M. (2007). Information Retrieval for Music and Motion. Springer
		// Berlin Heidelberg. https://doi.org/10.1007/978-3-540-74048-3

		// Ratanamahatana, C., & Keogh, E. (2004). Everything you know about dynamic
		// time warping is wrong.

		// Wu, R., & Keogh, E. J. (2020). FastDTW is approximate and Generally Slower
		// than the Algorithm it Approximates. IEEE Transactions on Knowledge and Data
		// Engineering, 1–1. https://doi.org/10.1109/TKDE.2020.3033752

		auto matrix = this->m_factory.template make<0>(len_s, len_t);
		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {
			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

				typename Global<Index, Value>::template AccumulatorFactory<
					Direction, ComputationGoal>::Accumulator acc;

				acc.set(
					values(u - 1, v - 1),
					u - 1, v - 1);

				acc.push(
					values(u - 1, v),
					u - 1, v);

				acc.push(
					values(u, v - 1),
					u, v - 1);

				acc.add(pairwise(u, v));

				acc.write(
					values(u, v),
					xt::view(traceback, u, v, xt::all()));
			}
		}
	}
};

} // namespace pyalign

#endif // __PYALIGN_SOLVER__
