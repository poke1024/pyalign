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

namespace goal {
	struct score { // compute only score
	};

	template<typename PathGoal>
	struct alignment { // compute full traceback
		typedef PathGoal path_goal;
	};

	namespace path {
		struct one { // track one optimal path
		};

		struct all { // track all optimal paths
		};
	}

	typedef alignment<path::one> one_alignment;
	typedef alignment<path::all> all_alignments;
}

namespace direction {
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
}

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

class AlgorithmMetaData {
	const std::string m_name;
	const std::string m_runtime;
	const std::string m_memory;

public:
	AlgorithmMetaData(const char *p_name, const char *p_runtime, const char *p_memory) :
		m_name(p_name), m_runtime(p_runtime), m_memory(p_memory) {
	}

	const std::string &name() const {
		return m_name;
	}

	const std::string &runtime() const {
		return m_runtime;
	}

	const std::string &memory() const {
		return m_memory;
	}
};

typedef std::shared_ptr<AlgorithmMetaData> AlgorithmMetaDataRef;

template<typename Value>
using GapTensorFactory = std::function<xt::xtensor<Value, 1>(size_t)>;


template<typename Index>
struct traceback_1 {
	static constexpr bool multiple = false;

	xt::xtensor_fixed<Index, xt::xshape<2>> uv;

	inline void init(const Index u, const Index v) {
		uv(0) = u;
		uv(1) = v;
	}

	inline void init(const traceback_1 &tb) {
		uv = tb.uv;
	}

	inline void push(const Index u, const Index v) {
		uv(0) = u;
		uv(1) = v;
	}

	inline void push(const traceback_1 &tb) {
		uv = tb.uv;
	}

	inline Index u() const {
		return uv(0);
	}

	inline Index v() const {
		return uv(1);
	}
};

template<typename Index>
struct traceback_n {
	static constexpr bool multiple = true;

	typedef xt::xtensor_fixed<Index, xt::xshape<2>> Pt;

	std::vector<Pt> pts;

	inline void init(const Index u, const Index v) {
		pts.clear();
		pts.emplace_back(Pt{u, v});
	}

	inline void init(const traceback_n &tb) {
		pts = tb.pts;
	}

	inline void push(const Index u, const Index v) {
		pts.emplace_back(Pt{u, v});
	}

	inline void push(const traceback_n &tb) {
		for (const auto &pt : tb.pts) {
			pts.push_back(pt);
		}
	}

	inline Index u() const {
		return pts.empty() ? -1 : pts[0](0);
	}

	inline Index v() const {
		return pts.empty() ? -1 : pts[0](1);
	}
};

template<typename Value, typename Index, template<typename> class Traceback>
struct cell_type {
	typedef Value value_type;
	typedef Index index_type;
	typedef Traceback<Index> traceback_type;
};


template<typename CellType, int LayerCount, int Layer>
class Matrix;

template<typename CellType, int LayerCount>
class MatrixFactory {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;
	typedef typename CellType::traceback_type Traceback;

protected:
	friend class Matrix<CellType, LayerCount, 0>;
	friend class Matrix<CellType, LayerCount, 1>;
	friend class Matrix<CellType, LayerCount, 2>;

	struct Data {
		xt::xtensor<Value, 3> values;
		xt::xtensor<Traceback, 3> traceback;
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
			m_max_len_t + 1
		});
		m_data->best_column.resize({
			m_max_len_s + 1
		});
	}

	template<int Layer>
	inline Matrix<CellType, LayerCount, Layer> make(
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

	struct all_layers_accessor {
		Data &m_data;

		inline auto values(const Index len_s, const Index len_t) const {
			return xt::view(
				m_data.values,
				xt::all(), xt::range(0, len_s + 1), xt::range(0, len_t + 1));
		}

		inline auto traceback(const Index len_s, const Index len_t) const {
			return xt::view(
				m_data.traceback,
				xt::all(), xt::range(0, len_s + 1), xt::range(0, len_t + 1));
		}
	};

	all_layers_accessor all_layers() const {
		return all_layers_accessor{*m_data.get()};
	}
};

template<ssize_t i0, ssize_t j0, typename View>
struct shifted_xview {
	View v;

	typename View::value_type operator()(const ssize_t i, const ssize_t j) const {
		return v(i + i0, j + j0);
	}

	typename View::reference operator()(const ssize_t i, const ssize_t j) {
		return v(i + i0, j + j0);
	}
};

template<ssize_t i0, ssize_t j0, typename View>
shifted_xview<i0, j0, View> shift_xview(View &&v) {
	return shifted_xview<i0, j0, View>{v};
}

template<typename CellType, int LayerCount, int Layer>
class Matrix {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;

private:
	const MatrixFactory<CellType, LayerCount> &m_factory;
	const Index m_len_s;
	const Index m_len_t;

public:
	inline Matrix(
		const MatrixFactory<CellType, LayerCount> &factory,
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
		return shift_xview<i0, j0>(xt::view(
			m_factory.m_data->values, Layer, xt::all(), xt::all()));
	}

	template<int i0, int j0>
	inline auto traceback_n() const {
		return shift_xview<i0, j0>(xt::view(
			m_factory.m_data->traceback, Layer, xt::all(), xt::all()));
	}

	struct assume_non_negative_indices {
	};

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

template<typename CellType, int LayerCount>
template<int Layer>
inline Matrix<CellType, LayerCount, Layer> MatrixFactory<CellType, LayerCount>::make(
	const Index len_s, const Index len_t) const {

	static_assert(Layer < LayerCount, "layer index exceeds layer count");
	check_size_against_max(len_s, m_max_len_s);
	check_size_against_max(len_t, m_max_len_t);
	return Matrix<CellType, LayerCount, Layer>(*this, len_s, len_t);
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


template<typename Direction, typename CellType>
class Accumulator {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::traceback_type Traceback;

private:
	Value m_val;

	Value &m_cell_val;

public:
	inline Accumulator(Value &p_val, Traceback &p_tb) : m_cell_val(p_val) {
	}

	inline void init(
		const Value val,
		const Index u,
		const Index v) {

		m_val = val;
	}

	inline void push(
		const Value val,
		const Index u,
		const Index v) {

		if (Direction::is_improvement(val, m_val)) {
			m_val = val;
		}
	}

	inline void push(
		const Value val,
		const Traceback &tb) {

		if (Direction::is_improvement(val, m_val)) {
			m_val = val;
		}
	}

	inline void add(const float val) {
		m_val += val;
	}

	inline void done() const {
		m_cell_val = m_val;
	}
};

template<typename Direction, typename CellType>
class TracingAccumulator {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::traceback_type Traceback;

private:
	Value m_val;

	Value &m_cell_val;
	Traceback &m_cell_tb;

public:
	inline TracingAccumulator(Value &p_val, Traceback &p_tb) :
		m_cell_val(p_val), m_cell_tb(p_tb) {
	}

	inline void init(
		const Value val,
		const Index u,
		const Index v) {

		m_val = val;
		m_cell_tb.init(u, v);
	}

	inline void push(
		const Value val,
		const Index u,
		const Index v) {

		if (Direction::is_improvement(val, m_val)) {
			m_val = val;
			m_cell_tb.init(u, v);
		} else if (Traceback::multiple && val == m_val) {
			m_cell_tb.push(u, v);
		}
	}

	inline void push(
		const Value val,
		const Traceback &tb) {

		if (Direction::is_improvement(val, m_val)) {
			m_val = val;
			m_cell_tb.init(tb);
		} else if (Traceback::multiple && val == m_val) {
			m_cell_tb.push(tb);
		}
	}

	inline void add(const float val) {
		m_val += val;
	}

	inline void done() const {
		m_cell_val = m_val;
	}
};


template<typename CellType, typename Strategy, typename Matrix, typename Path>
class Traceback {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

private:
	const Strategy m_strategy;
	Matrix &m_matrix;
	Path &m_path;
	std::optional<std::pair<Index, Index>> m_seed;

public:
	inline Traceback(
		Strategy &&p_strategy,
		Matrix &p_matrix,
		Path &p_path) :
		m_strategy(p_strategy),
		m_matrix(p_matrix),
		m_path(p_path) {

		m_seed = m_strategy.seed();
	}

	inline std::optional<Value> next() {
		if (!m_seed.has_value()) {
			return std::optional<Value>();
		}

		const auto values = m_matrix.template values<1, 1>();

		const auto initial_uv = m_seed.value();
		m_seed.reset();

		Index u = std::get<0>(initial_uv);
		Index v = std::get<1>(initial_uv);
		const auto best_val = values(u, v);

		if (typename Strategy::template trace_path<Path>()) {
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();
			m_path.begin(len_s, len_t);

			const auto traceback = m_matrix.template traceback<1, 1>();

			while (m_strategy.cont(u, v)) {
				const Index last_u = u;
				const Index last_v = v;

				const auto &t = traceback(u, v);
				u = t.u();
				v = t.v();

				m_path.step(last_u, last_v, u, v);
			}
		}

		return m_strategy.path_val(best_val, u, v);
	}
};

template<typename Locality, typename Matrix, typename Path>
inline Traceback<
	typename Locality::cell_type,
	typename Locality::template TracebackStrategy<Matrix>,
	Matrix, Path>
	make_traceback(
		const Locality &p_locality,
		Matrix &p_matrix,
		Path &p_path) {

	return Traceback<
		typename Locality::cell_type,
		typename Locality::template TracebackStrategy<Matrix>,
		Matrix, Path>(

		typename Locality::template TracebackStrategy<Matrix>(p_locality, p_matrix),
		p_matrix,
		p_path
	);
}

struct LocalInitializers {
	float zero;
};

template<typename CellType>
class Local {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef CellType cell_type;

	static constexpr bool is_global() {
		return false;
	}

private:
	const Value m_zero;

public:
	template<typename Direction, typename Goal>
	struct AccumulatorFactory {
		typedef TracingAccumulator<Direction, CellType> Accumulator;
	};

	inline Local(const LocalInitializers &p_init) : m_zero(p_init.zero) {
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

	inline Value zero() const {
		return m_zero;
	}

	template<typename Matrix>
	class TracebackStrategy {
		const float m_zero;
		Matrix &m_matrix;

	public:
		inline TracebackStrategy(const Local<CellType> &p_locality, Matrix &p_matrix) :
			m_zero(p_locality.zero()),
			m_matrix(p_matrix) {
		}

		template<typename Path>
		struct trace_path {
			constexpr operator bool() const {
				return true;
			}
		};

		inline std::optional<std::pair<Index, Index>> seed() const {
			const auto values = m_matrix.template values_n<1, 1>();
			auto best_column = m_matrix.best_column();

			best_column = xt::argmax(m_matrix.template values<1, 1>(), 1);

			Value score = m_zero;
			Index best_u = 0, best_v = 0;

			const auto len_s = m_matrix.len_s();
			for (Index u = 0; u < len_s; u++) {
				const Index v = best_column(u);
				const Value s = values(u, v);
				if (s > score) {
					score = s;
					best_u = u;
					best_v = v;
				}
			}

			if (score > m_zero) {
				return std::make_pair(best_u, best_v);
			} else {
				return std::optional<std::pair<Index, Index>>();
			}
		}

		inline bool cont(const Index u, const Index v) const {
			const auto values = m_matrix.template values_n<1, 1>();
			return u >= 0 && v >= 0 && values(u, v) > m_zero;
		}

		inline Value path_val(
			const Value best_val,
			const Index u,
			const Index v) const {

			if (u >= 0 && v >= 0) {
				const auto values = m_matrix.template values_n<1, 1>();
				return best_val - values(u, v);
			} else {
				return best_val;
			}
		}
	};
};

struct GlobalInitializers {
};

template<typename CellType>
class Global {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef CellType cell_type;

	static constexpr bool is_global() {
		return true;
	}

public:
	template<typename Direction, typename Goal>
	struct AccumulatorFactory {
	};

	template<typename Direction>
	struct AccumulatorFactory<Direction, goal::score> {
		typedef Accumulator<Direction, CellType> Accumulator;
	};

	template<typename Direction, typename PathGoal>
	struct AccumulatorFactory<Direction, goal::alignment<PathGoal>> {
		typedef TracingAccumulator<Direction, CellType> Accumulator;
	};

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		if (p_vector.size() != p_gap_cost.size()) {
			throw std::runtime_error("size mismatch in init_border_case");
		}

		p_vector = p_gap_cost;
		//xt::view(p_gap_cost, xt::range(0, p_vector.size()));
	}

	inline Global(const GlobalInitializers&) {
	}

	template<typename Accumulator>
	inline void update_acc(Accumulator &) const {
	}

	template<typename Matrix>
	class TracebackStrategy {
		Matrix &m_matrix;

	public:
		inline TracebackStrategy(
			const Global<CellType> &p_locality,
			Matrix &p_matrix) :
			m_matrix(p_matrix) {
		}

		template<typename Path>
		struct trace_path {
			constexpr operator bool() const {
				return true;
			}
		};

		template<>
		struct trace_path<build_nothing> {
			constexpr operator bool() const {
				return false;
			}
		};

		inline std::optional<std::pair<Index, Index>> seed() const {
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();
			return std::make_pair(len_s - 1, len_t - 1);
		}

		inline bool cont(const Index u, const Index v) const {
			const auto values = m_matrix.template values_n<1, 1>();
			return u >= 0 && v >= 0;
		}

		inline Value path_val(
			const Value best_val,
			const Index u,
			const Index v) const {

			return best_val;
		}
	};
};


struct SemiglobalInitializers {
};

template<typename CellType>
class Semiglobal {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef CellType cell_type;

	static constexpr bool is_global() {
		return false;
	}

public:
	template<typename Direction, typename Goal>
	struct AccumulatorFactory {
	};

	template<typename Direction>
	struct AccumulatorFactory<Direction, goal::score> {
		typedef Accumulator<Direction, CellType> Accumulator;
	};

	template<typename Direction, typename PathGoal>
	struct AccumulatorFactory<Direction, goal::alignment<PathGoal>> {
		typedef TracingAccumulator<Direction, CellType> Accumulator;
	};

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		p_vector.fill(0);
	}

	inline Semiglobal(const SemiglobalInitializers&) {
	}

	template<typename Accumulator>
	inline void update_acc(Accumulator &) const {
	}

	template<typename Matrix>
	class TracebackStrategy {
		Matrix &m_matrix;

	public:
		inline TracebackStrategy(
			const Semiglobal<CellType> &p_locality,
			Matrix &p_matrix) :
			m_matrix(p_matrix) {
		}

		template<typename Path>
		struct trace_path {
			constexpr operator bool() const {
				return true;
			}
		};

		template<>
		struct trace_path<build_nothing> {
			constexpr operator bool() const {
				return false;
			}
		};

		inline std::optional<std::pair<Index, Index>> seed() const {
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			const auto values = m_matrix.template values_n<1, 1>();

			const Index last_row = len_s - 1;
			const Index last_col = len_t - 1;

			const auto values_non_neg_ij = m_matrix.template values<1, 1>();
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

			return std::make_pair(u, v);
		}

		inline bool cont(const Index u, const Index v) const {
			const auto values = m_matrix.template values_n<1, 1>();
			return u >= 0 && v >= 0;
		}

		inline Value path_val(
			const Value best_val,
			const Index u,
			const Index v) const {

			return best_val;
		}
	};
};

template<typename CellType>
class Solution {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::traceback_type Traceback;
	typedef CellType cell_type;

	xt::xtensor<Value, 3> m_values;
	xt::xtensor<Traceback, 3> m_traceback;
	xt::xtensor<Index, 2> m_path;
	Value m_score;
	AlgorithmMetaDataRef m_algorithm;

	const auto &values() const {
		return m_values;
	}

	const auto traceback() const {
		const size_t len_k = m_traceback.shape(0);
		const size_t len_s = m_traceback.shape(1);
		const size_t len_t = m_traceback.shape(2);

		xt::xtensor<Index, 4> traceback;
		traceback.resize({
			len_k, len_s, len_t, 2
		});
		for (size_t k = 0; k < len_k; k++) {
			for (size_t i = 0; i < len_s; i++) {
				for (size_t j = 0; j < len_t; j++) {
					traceback(k, i, j, 0) = m_traceback(k, i, j).u();
					traceback(k, i, j, 1) = m_traceback(k, i, j).v();
				}
			}
		}
		return traceback;
	}

	const auto &path() const {
		return m_path;
	}

	const auto score() const {
		return m_score;
	}

	const auto &algorithm() const {
		return m_algorithm;
	}
};

template<typename CellType>
using SolutionRef = std::shared_ptr<Solution<CellType>>;


template<typename CellType, template<typename> class Locality, int LayerCount>
class Solver {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;

protected:
	const Locality<CellType> m_locality;
	MatrixFactory<CellType, LayerCount> m_factory;
	const AlgorithmMetaDataRef m_algorithm;

public:
	template<typename LocalityInitializers>
	inline Solver(
		const LocalityInitializers &p_locality_init,
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const AlgorithmMetaDataRef &p_algorithm) :

		m_locality(p_locality_init),
		m_factory(p_max_len_s, p_max_len_t),
		m_algorithm(p_algorithm) {
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

	inline Value worst_score() const {
		// FIXME: use std::numerical_limits<Value>::infinity()?
		return 0;
	}

	inline Value score(
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = m_factory.template make<0>(len_s, len_t);
		build_nothing nothing;
		auto tb = make_traceback(m_locality, matrix, nothing);
		const auto tb_val = tb.next();
		return tb_val.value_or(worst_score());
	}

	template<typename Alignment>
	inline Value alignment(
		const size_t len_s,
		const size_t len_t,
		Alignment &alignment) const {

		auto matrix = m_factory.template make<0>(len_s, len_t);
		build_alignment<Alignment> build(alignment);
		auto tb = make_traceback(m_locality, matrix, build);
		const auto tb_val = tb.next();
		return tb_val.value_or(worst_score());
	}

	template<typename Alignment>
	SolutionRef<CellType> solution(
		const size_t len_s,
		const size_t len_t,
		Alignment &alignment) const {

		const SolutionRef<CellType> solution =
			std::make_shared<Solution<CellType>>();

		solution->m_values = m_factory.all_layers().values(len_s, len_t);
		solution->m_traceback = m_factory.all_layers().traceback(len_s, len_t);

		auto build = build_multiple<build_path<Index>, build_alignment<Alignment>>(
			build_path<Index>(), build_alignment<Alignment>(alignment)
		);

		auto matrix = m_factory.template make<0>(len_s, len_t);
		auto tb = make_traceback(m_locality, matrix, build);
		const auto tb_val = tb.next();
		solution->m_path = build.template get<0>().path();
		solution->m_score = tb_val.value_or(worst_score());

		solution->m_algorithm = m_algorithm;

		return solution;
	}
};

template<typename CellType, template<typename> class Locality, int LayerCount>
using AlignmentSolver = Solver<CellType, Locality, LayerCount>;

template<typename Direction, typename CellType, template<typename> class Locality>
class LinearGapCostSolver final : public AlignmentSolver<CellType, Locality, 1> {
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
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

private:
	const Value m_gap_cost_s;
	const Value m_gap_cost_t;

public:
	typedef Value GapCostSpec;

	template<typename LocalityInitializers>
	inline LinearGapCostSolver(
		const LocalityInitializers &p_locality_init,
		const Value p_gap_cost_s,
		const Value p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		AlignmentSolver<CellType, Locality, 1>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>(
				Locality<CellType>::is_global() ? "Needleman-Wunsch": "Smith-Waterman",
				"n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto values = this->m_factory.template values<0>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		this->m_locality.init_border_case(
			xt::view(values, xt::all(), 0),
			xt::arange<Index>(0, p_max_len_s + 1) * p_gap_cost_s * gap_sgn);
		this->m_locality.init_border_case(
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

				typename Locality<CellType>::template AccumulatorFactory<
					Direction, ComputationGoal>::Accumulator acc(
						values(u, v), traceback(u, v));

				acc.init(
					values(u - 1, v - 1) + pairwise(u, v),
					u - 1, v - 1);

				acc.push(
					values(u - 1, v) + this->m_gap_cost_s * gap_sgn,
					u - 1, v);

				acc.push(
					values(u, v - 1) + this->m_gap_cost_t * gap_sgn,
					u, v - 1);

				this->m_locality.update_acc(acc);

				acc.done();
			}
		}
	}
};

template<typename Value>
struct AffineCost {
	// w(k) = u k + v

	Value u;
	Value v;

	inline AffineCost(Value p_u, Value p_v) : u(p_u), v(p_v) {
	}

	inline Value w1() const {
		return u + v; // i.e. w(1)
	}

	inline auto vector(size_t n) const {
		xt::xtensor<Value, 1> w = xt::linspace<Value>(0, (n - 1) * u, n) + v;
		w.at(0) = 0;
		return w;
	}
};

template<typename Direction, typename CellType, template<typename> class Locality>
class AffineGapCostSolver final : public AlignmentSolver<CellType, Locality, 3> {
public:
	// Gotoh, O. (1982). An improved algorithm for matching biological sequences.
	// Journal of Molecular Biology, 162(3), 705–708. https://doi.org/10.1016/0022-2836(82)90398-9

	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	typedef AffineCost<Value> Cost;

private:
	const Cost m_gap_cost_s;
	const Cost m_gap_cost_t;

public:
	template<typename LocalityInitializers>
	inline AffineGapCostSolver(
		const LocalityInitializers &p_locality_init,
		const Cost &p_gap_cost_s,
		const Cost &p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		AlignmentSolver<CellType, Locality, 3>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>("Gotoh", "n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto matrix_D = this->m_factory.template make<0>(p_max_len_s, p_max_len_t);
		auto matrix_P = this->m_factory.template make<1>(p_max_len_s, p_max_len_t);
		auto matrix_Q = this->m_factory.template make<2>(p_max_len_s, p_max_len_t);

		auto D = matrix_D.template values<0, 0>();
		auto P = matrix_P.template values<0, 0>();
		auto Q = matrix_Q.template values<0, 0>();

		const auto inf = std::numeric_limits<Value>::infinity() * (Direction::is_minimize() ? 1 : -1);

		xt::view(Q, xt::all(), 0).fill(inf);
		xt::view(P, 0, xt::all()).fill(inf);

		// setting D(m, 0) = P(m, 0) = w(m)
		this->m_locality.init_border_case(
			xt::view(D, xt::all(), 0),
			m_gap_cost_s.vector(p_max_len_s + 1));
		this->m_locality.init_border_case(
			xt::view(P, xt::all(), 0),
			m_gap_cost_s.vector(p_max_len_s + 1));

		// setting D(0, n) = Q(0, n) = w(n)
		this->m_locality.init_border_case(
			xt::view(D, 0, xt::all()),
			m_gap_cost_t.vector(p_max_len_t + 1));
		this->m_locality.init_border_case(
			xt::view(Q, 0, xt::all()),
			m_gap_cost_t.vector(p_max_len_t + 1));

		auto tb_P = matrix_P.template traceback<0, 0>();
		auto tb_Q = matrix_Q.template traceback<0, 0>();

		for (auto &e : xt::view(tb_P, 0, xt::all())) {
			e.init(-1, -1);
		}
		for (auto &e : xt::view(tb_P, xt::all(), 0)) {
			e.init(-1, -1);
		}
		for (auto &e : xt::view(tb_Q, 0, xt::all())) {
			e.init(-1, -1);
		}
		for (auto &e : xt::view(tb_Q, xt::all(), 0)) {
			e.init(-1, -1);
		}
	}

	template<typename ComputationGoal, typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix_D = this->m_factory.template make<0>(len_s, len_t);
		auto matrix_P = this->m_factory.template make<1>(len_s, len_t);
		auto matrix_Q = this->m_factory.template make<2>(len_s, len_t);

		auto D = matrix_D.template values_n<1, 1>();
		auto tb_D = matrix_D.template traceback_n<1, 1>();
		auto P = matrix_P.template values_n<1, 1>();
		auto tb_P = matrix_P.template traceback_n<1, 1>();
		auto Q = matrix_Q.template values_n<1, 1>();
		auto tb_Q = matrix_Q.template traceback_n<1, 1>();

		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		for (Index i = 0; static_cast<size_t>(i) < len_s; i++) {

			for (Index j = 0; static_cast<size_t>(j) < len_t; j++) {

				// Gotoh formula (4)
				{
					typename Locality<CellType>::template AccumulatorFactory<
						Direction, ComputationGoal>::Accumulator acc_P(
							P(i, j), tb_P(i, j));

					acc_P.init(D(i - 1, j) + m_gap_cost_s.w1() * gap_sgn, i - 1, j);
					acc_P.push(P(i - 1, j) + m_gap_cost_s.u * gap_sgn, tb_P(i - 1, j));
					acc_P.done();
				}

				// Gotoh formula (5)
				{
					typename Locality<CellType>::template AccumulatorFactory<
						Direction, ComputationGoal>::Accumulator acc_Q(
							Q(i, j), tb_Q(i, j));

					acc_Q.init(D(i, j - 1) + m_gap_cost_t.w1() * gap_sgn, i, j - 1);
					acc_Q.push(Q(i, j - 1) + m_gap_cost_t.u * gap_sgn, tb_Q(i, j - 1));
					acc_Q.done();
				}

				// Gotoh formula (1)
				{
					typename Locality<CellType>::template AccumulatorFactory<
						Direction, ComputationGoal>::Accumulator acc_D(
							D(i, j), tb_D(i, j));

					acc_D.init(D(i - 1, j - 1) + pairwise(i, j), i - 1, j - 1);
					acc_D.push(P(i, j), tb_P(i, j));
					acc_D.push(Q(i, j), tb_Q(i, j));
					this->m_locality.update_acc(acc_D);
					acc_D.done();
				}
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

template<typename Direction, typename CellType, template<typename> class Locality>
class GeneralGapCostSolver final : public AlignmentSolver<CellType, Locality, 1> {
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
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

private:
	const xt::xtensor<Value, 1> m_gap_cost_s;
	const xt::xtensor<Value, 1> m_gap_cost_t;

public:
	typedef GapTensorFactory<Value> GapCostSpec;

	template<typename LocalityInitializers>
	inline GeneralGapCostSolver(
		const LocalityInitializers &p_locality_init,
		const GapTensorFactory<Value> &p_gap_cost_s,
		const GapTensorFactory<Value> &p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		AlignmentSolver<CellType, Locality, 1>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>("Waterman-Smith-Beyer", "n^3", "n^2")),
		m_gap_cost_s(p_gap_cost_s(p_max_len_s + 1)),
		m_gap_cost_t(p_gap_cost_t(p_max_len_t + 1)) {

		check_gap_tensor_shape(m_gap_cost_s, p_max_len_s + 1);
		check_gap_tensor_shape(m_gap_cost_t, p_max_len_t + 1);

		auto values = this->m_factory.template values<0>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		this->m_locality.init_border_case(
			xt::view(values, xt::all(), 0),
			m_gap_cost_s * gap_sgn);

		this->m_locality.init_border_case(
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

				typename Locality<CellType>::template AccumulatorFactory<
					Direction, ComputationGoal>::Accumulator acc(
						values(u, v), traceback(u, v));

				acc.init(
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

				acc.done();
			}
		}
	}
};

template<typename Direction, typename CellType>
class DynamicTimeSolver final : public Solver<CellType, Global, 1> {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	inline DynamicTimeSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		Solver<CellType, Global, 1>(
			GlobalInitializers(),
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>("DTW", "n^2", "n^2")) {

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

				typename Global<CellType>::template AccumulatorFactory<
					Direction, ComputationGoal>::Accumulator acc(
						values(u, v), traceback(u, v));

				acc.init(
					values(u - 1, v - 1),
					u - 1, v - 1);

				acc.push(
					values(u - 1, v),
					u - 1, v);

				acc.push(
					values(u, v - 1),
					u, v - 1);

				acc.add(pairwise(u, v));

				acc.done();
			}
		}
	}
};

} // namespace pyalign

#endif // __PYALIGN_SOLVER__
