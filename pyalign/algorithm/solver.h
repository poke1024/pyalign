#ifndef __PYALIGN_SOLVER__
#define __PYALIGN_SOLVER__

#define XTENSOR_USE_XSIMD 1

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xsort.hpp>

#include <stack>

namespace pyalign {

namespace goal {
	struct optimal_score { // compute only optimal score
		struct path_goal {
		};
	};

	template<typename PathGoal>
	struct alignment { // compute full traceback
		typedef PathGoal path_goal;
	};

	namespace path {
		namespace optimal {
			struct one { // generate one optimal path
			};

			struct all { // generate all optimal paths
			};
		}

		/*struct all { // generate all paths
		};*/
	}

	typedef alignment<path::optimal::one> one_optimal_alignment;
	typedef alignment<path::optimal::all> all_optimal_alignments;
}

namespace direction {
	struct maximize {
		template<typename A, typename B>
		static inline auto opt(A &&a, B &&b) {
			return xt::maximum(a, b);
		}

		template<typename A, typename B>
		static inline auto opt_q(A &&a, B &&b) {
			return xt::greater(a, b);
		}

		template<typename Value>
		static inline bool is_opt(Value a, Value b) {
			return a > b;
		}

		static constexpr bool is_minimize() {
			return false;
		}

		template<typename Value>
		static constexpr Value worst_val() {
			return -std::numeric_limits<Value>::infinity();
		}
	};

	struct minimize {
		template<typename A, typename B>
		static inline auto opt(A &&a, B &&b) {
			return xt::minimum(a, b);
		}

		template<typename A, typename B>
		static inline auto opt_q(A &&a, B &&b) {
			return xt::less(a, b);
		}

		template<typename Value>
		static inline bool is_opt(Value a, Value b) {
			return a < b;
		}

		static constexpr bool is_minimize() {
			return true;
		}

		template<typename Value>
		static constexpr Value worst_val() {
			return std::numeric_limits<Value>::infinity();
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
constexpr inline Index no_traceback() {
	return std::numeric_limits<Index>::min();
}

template<typename CellType>
inline auto no_traceback_vec() {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	IndexVec v;
	v.fill(std::numeric_limits<Index>::min());
	return v;
}

template<typename CellType>
struct traceback_1 {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::mask_vec_type MaskVec;

private:
	struct {
		IndexVec u;
		IndexVec v;
	} uv;

public:
	static constexpr bool max_degree_1 = true;

	struct Single {
		Index _u;
		Index _v;

		inline Index size() const {
			return 1;
		}

		inline Index u(const size_t i) const {
			return _u;
		}

		inline Index v(const size_t i) const {
			return _v;
		}
	};

	inline Single to_single(const int batch_index) const {
		return Single{uv.u[batch_index], uv.v[batch_index]};
	}

	inline void clear() {
		uv.u = no_traceback_vec<CellType>();
		uv.v = no_traceback_vec<CellType>();
	}

	inline void init(const Index p_u, const Index p_v) {
		uv.u.fill(p_u);
		uv.v.fill(p_v);
	}

	inline void init(const Index p_u, const Index p_v, const MaskVec &mask) {
		const IndexVec u = xt::full_like(IndexVec(), p_u);
		const IndexVec v = xt::full_like(IndexVec(), p_v);
		uv.u = xt::where(mask, u, uv.u);
		uv.v = xt::where(mask, v, uv.v);
	}

	inline void init(const traceback_1 &tb, const MaskVec &mask) {
		uv.u = xt::where(mask, tb.uv.u, uv.u);
		uv.v = xt::where(mask, tb.uv.v, uv.v);
	}

	inline void push(const Index p_u, const Index p_v, const MaskVec &mask) {
		const IndexVec u = xt::full_like(IndexVec(), p_u);
		const IndexVec v = xt::full_like(IndexVec(), p_v);
		uv.u = xt::where(mask, u, uv.u);
		uv.v = xt::where(mask, v, uv.v);
	}

	inline void push(const traceback_1 &tb, const MaskVec &mask) {
		uv.u = xt::where(mask, tb.uv.u, uv.u);
		uv.v = xt::where(mask, tb.uv.v, uv.v);
	}

	inline Index u(const int batch_index, const size_t i) const {
		return uv.u(batch_index);
	}

	inline Index v(const int batch_index, const size_t i) const {
		return uv.v(batch_index);
	}

	inline Index size(const int batch_index) const {
		return uv.u(batch_index) != no_traceback<Index>() ? 1 : 0;
	}
};

template<typename CellType>
struct traceback_n {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::mask_vec_type MaskVec;

	static constexpr int BatchSize = CellType::batch_size;

	struct Pt {
		Index u;
		Index v;
	};

	std::vector<Pt> pts[BatchSize];

public:
	static constexpr bool max_degree_1 = false;

	struct Single {
		std::vector<Pt> pts;

		inline Index size() const {
			return pts.size();
		}

		inline Index u(const size_t i) const {
			return pts[i].u;
		}

		inline Index v(const size_t i) const {
			return pts[i].v;
		}
	};

	inline Single to_single(const int batch_index) const {
		return Single{pts[batch_index]};
	}

	inline void clear() {
		for (int i = 0; i < BatchSize; i++) {
			pts[i].clear();
		}
	}

	inline void init(const Index p_u, const Index p_v) {
		for (int i = 0; i < BatchSize; i++) {
			pts[i].clear();
			pts[i].emplace_back(Pt{p_u, p_v});
		}
	}

	inline void init(const Index p_u, const Index p_v, const MaskVec &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			pts[i].clear();
			pts[i].emplace_back(Pt{p_u, p_v});
		}
	}

	inline void init(const traceback_n &tb, const MaskVec &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			pts[i] = tb.pts[i];
		}
	}

	inline void push(const Index p_u, const Index p_v, const MaskVec &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			pts[i].emplace_back(Pt{p_u, p_v});
		}
	}

	inline void push(const traceback_n &tb, const MaskVec &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			for (const auto &pt : tb.pts[i]) {
				pts[i].push_back(pt);
			}
		}
	}

	inline Index u(const int batch_index, const size_t i) const {
		return i < pts[batch_index].size() ?
			pts[batch_index][i].u :
			no_traceback<Index>();
	}

	inline Index v(const int batch_index, const size_t i) const {
		return i < pts[batch_index].size() ?
			pts[batch_index][i].v :
			no_traceback<Index>();
	}

	inline Index size(const int batch_index) const {
		return pts[batch_index].size();
	}
};

struct no_batch {
	template<typename Value, typename Index>
	struct size {
		static constexpr int s = 1;
	};
};

template<int S>
struct custom_batch_size {
	template<typename Value, typename Index>
	struct size {
		static constexpr int s = S;
	};
};

struct machine_batch_size {
	template<typename Value, typename Index>
	struct size {
#if __AVX512F__
		static constexpr int s = 512 / (std::max(sizeof(Value), sizeof(Index)) * 8);
#elif __AVX2__
		static constexpr int s = 256 / (std::max(sizeof(Value), sizeof(Index)) * 8);
#elif __SSE__
		static constexpr int s = 128 / (std::max(sizeof(Value), sizeof(Index)) * 8);
#else
		static constexpr int s = 1;
#endif
	};
};

template<typename Value, typename Index, typename Batch = no_batch>
struct cell_type {
	typedef Value value_type;
	typedef Index index_type;

	static constexpr int batch_size = Batch::template size<Value, Index>::s;

	typedef xt::xtensor_fixed<Value, xt::xshape<batch_size>> value_vec_type;
	typedef xt::xtensor_fixed<Index, xt::xshape<batch_size>> index_vec_type;
	typedef xt::xtensor_fixed<bool, xt::xshape<batch_size>> mask_vec_type;
};

template<typename Goal, typename Direction>
struct problem_type {
	typedef Goal goal_type;
	typedef Direction direction_type;
};


template<typename PathGoal, typename CellType>
struct traceback_cell_type_factory {
};

template<typename CellType>
struct traceback_cell_type_factory<goal::optimal_score::path_goal, CellType> {
	typedef traceback_1<CellType> traceback_cell_type;
};

template<typename CellType>
struct traceback_cell_type_factory<goal::path::optimal::one, CellType> {
	typedef traceback_1<CellType> traceback_cell_type;
};

template<typename CellType>
struct traceback_cell_type_factory<goal::path::optimal::all, CellType> {
	typedef traceback_n<CellType> traceback_cell_type;
};

template<typename CellType, typename ProblemType>
using traceback_type = typename traceback_cell_type_factory<
	typename ProblemType::goal_type::path_goal,
	CellType>::traceback_cell_type;


template<typename CellType, typename ProblemType, int LayerCount, int Layer>
class Matrix;

template<typename CellType, typename ProblemType, int LayerCount>
class MatrixFactory {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename CellType::index_vec_type IndexVec;
	typedef traceback_type<CellType, ProblemType> Traceback;

protected:
	friend class Matrix<CellType, ProblemType, LayerCount, 0>;
	friend class Matrix<CellType, ProblemType, LayerCount, 1>;
	friend class Matrix<CellType, ProblemType, LayerCount, 2>;

	struct Data {
		xt::xtensor<ValueVec, 3> values;
		xt::xtensor<Traceback, 3> traceback;
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

		for (int k = 0; k < LayerCount; k++) {
			for (size_t i = 0; i < m_max_len_s + 1; i++) {
				m_data->traceback(k, i, 0).clear();
			}
			for (size_t j = 0; j < m_max_len_t + 1; j++) {
				m_data->traceback(k, 0, j).clear();
			}
		}
	}

	template<int Layer>
	inline Matrix<CellType, ProblemType, LayerCount, Layer> make(
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

template<typename CellType, typename ProblemType, int LayerCount, int Layer>
class Matrix {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;

private:
	const MatrixFactory<CellType, ProblemType, LayerCount> &m_factory;
	const Index m_len_s;
	const Index m_len_t;

public:
	inline Matrix(
		const MatrixFactory<CellType, ProblemType, LayerCount> &factory,
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
};

template<typename CellType, typename ProblemType, int LayerCount>
template<int Layer>
inline Matrix<CellType, ProblemType, LayerCount, Layer>
MatrixFactory<CellType, ProblemType, LayerCount>::make(
	const Index len_s, const Index len_t) const {

	static_assert(Layer < LayerCount, "layer index exceeds layer count");
	check_size_against_max(len_s, m_max_len_s);
	check_size_against_max(len_t, m_max_len_t);
	return Matrix<CellType, ProblemType, LayerCount, Layer>(*this, len_s, len_t);
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


template<typename CellType, typename ProblemType>
class Accumulator {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename ProblemType::direction_type Direction;
	typedef traceback_type<CellType, ProblemType> Traceback;

public:
	struct cont {
		ValueVec &m_val;

		inline auto push(
			const ValueVec &val,
			const Index u,
			const Index v) {

			m_val = Direction::opt(val, m_val);
			return cont{m_val};
		}

		inline auto push(
			const ValueVec &val,
			const Traceback &tb) {

			m_val = Direction::opt(val, m_val);
			return cont{m_val};
		}

		template<typename F>
		inline auto push_many(const F &f) {
			f(cont{m_val});
			return cont{m_val};
		}

		inline auto add(const ValueVec &val) {
			m_val += val;
			return cont{m_val};
		}

		inline void done() const {
		}
	};

	struct init {
	protected:
		friend class Accumulator;

		inline explicit init(ValueVec &p_val) : m_val(p_val) {
		}

		init(const init&) = delete;
		init& operator=(init const&) = delete;

	public:
		ValueVec &m_val;

		inline auto push(
			const ValueVec &val,
			const Index u,
			const Index v) {

			m_val = val;
			return cont{m_val};
		}

		inline auto push(
			const ValueVec &val,
			const Traceback &tb) {

			m_val = val;
			return cont{m_val};
		}
	};

	static inline auto create(ValueVec &p_val, Traceback &p_tb) {
		return init{p_val};
	}
};

template<typename CellType, typename ProblemType>
class TracingAccumulator {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename CellType::mask_vec_type MaskVec;
	typedef typename ProblemType::direction_type Direction;
	typedef traceback_type<CellType, ProblemType> Traceback;

	static constexpr int BatchSize = CellType::batch_size;

	struct cont {
		ValueVec &m_val;
		Traceback &m_tb;

		inline auto push(
			const ValueVec &val,
			const Index u,
			const Index v) {

			if (BatchSize == 1 && Traceback::max_degree_1) {
				if (Direction::is_opt(val(0), m_val(0))) {
					m_val = val;
					m_tb.init(u, v);
				}
			} else {
				m_tb.init(u, v, Direction::opt_q(val, m_val));
				if (!Traceback::max_degree_1) {
					m_tb.push(u, v, xt::equal(val, m_val));
				}

				m_val = Direction::opt(val, m_val);
			}

			return cont{m_val, m_tb};
		}

		inline auto push(
			const ValueVec &val,
			const Traceback &tb) {

			if (BatchSize == 1 && Traceback::max_degree_1) {
				if (Direction::is_opt(val(0), m_val(0))) {
					m_val = val;
					m_tb.init(tb, xt::ones<bool>({BatchSize}));
				}
			} else {
				m_tb.init(tb, Direction::opt_q(val, m_val));
				if (!Traceback::max_degree_1) {
					m_tb.push(tb, xt::equal(val, m_val));
				}

				m_val = Direction::opt(val, m_val);
			}

			return cont{m_val, m_tb};
		}

		template<typename F>
		inline auto push_many(const F &f) {
			f(cont{m_val, m_tb});
			return cont{m_val, m_tb};
		}

		inline auto add(const ValueVec &val) {
			m_val += val;
			return cont{m_val, m_tb};
		}

		inline void done() const {
		}
	};

	struct init {
	protected:
		friend class TracingAccumulator;

		inline explicit init(ValueVec &p_val, Traceback &p_tb) :
			m_val(p_val), m_tb(p_tb) {
		}

		init(const init&) = delete;
		init& operator=(init const&) = delete;

	public:
		ValueVec &m_val;
		Traceback &m_tb;

		inline auto push(
			const ValueVec &val,
			const Index u,
			const Index v) {

			m_val = val;
			m_tb.init(u, v);
			return cont{m_val, m_tb};
		}

		inline auto push(
			const ValueVec &val,
			const Traceback &tb) {

			m_val = val;
			m_tb.init(tb, xt::ones<bool>({BatchSize}));
			return cont{m_val, m_tb};
		}
	};

	static inline auto create(ValueVec &p_val, Traceback &p_tb) {
		return init{p_val, p_tb};
	}
};

template<typename T>
class Stack1 {
	std::optional<T> m_data;

public:
	inline void push(T &&v) {
		m_data = v;
	}

	inline bool empty() const {
		return !m_data.has_value();
	}

	inline const T& top() const {
		return m_data.value();
	}

	inline void pop() {
		m_data.reset();
	}
};

template<bool Multiple, typename CellType, typename ProblemType, typename Strategy, typename Matrix>
class TracebackIterator;

template<typename CellType, typename ProblemType, typename Strategy, typename Matrix>
class TracebackIterator<false, CellType, ProblemType, Strategy, Matrix> {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	constexpr static int BatchSize = CellType::batch_size;

private:
	const Strategy m_strategy;
	Matrix &m_matrix;
	Stack1<std::pair<Index, Index>> m_seed[BatchSize];

public:
	inline TracebackIterator(
		Strategy &&p_strategy,
		Matrix &p_matrix) :
		m_strategy(p_strategy),
		m_matrix(p_matrix) {

		m_strategy.seeds().generate(m_seed);
	}

	template<typename Path>
	inline std::optional<Value> next(const int p_batch_index, Path &p_path) {
		auto &seed = m_seed[p_batch_index];

		if (seed.empty()) {
			return std::optional<Value>();
		}

		const auto values = m_matrix.template values<1, 1>();

		const auto initial_uv = seed.top();
		seed.pop();

		Index u = std::get<0>(initial_uv);
		Index v = std::get<1>(initial_uv);
		const auto best_val = values(u, v)(p_batch_index);

		if (m_strategy.has_trace()) { // && m_path.wants_path()
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();
			p_path.begin(len_s, len_t);

			const auto traceback = m_matrix.template traceback<1, 1>();

			while (m_strategy.continue_traceback(u, v, values(u, v)(p_batch_index))) {
				const Index last_u = u;
				const Index last_v = v;

				const auto &t = traceback(u, v);
				u = t.u(p_batch_index, 0);
				v = t.v(p_batch_index, 0);

				p_path.step(last_u, last_v, u, v);
			}
		}

		return best_val;
	}
};

template<typename CellType, typename ProblemType, typename Strategy, typename Matrix>
class TracebackIterator<true, CellType, ProblemType, Strategy, Matrix> {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	constexpr static int BatchSize = CellType::batch_size;

private:
	const Strategy m_strategy;
	Matrix &m_matrix;
	std::stack<std::pair<Index, Index>> m_uvs[BatchSize];

public:
	inline TracebackIterator(
		Strategy &&p_strategy,
		Matrix &p_matrix) :
		m_strategy(p_strategy),
		m_matrix(p_matrix) {

		m_strategy.seeds().generate(m_uvs);
	}

	template<typename Path>
	inline std::optional<Value> next(const int p_batch_index, Path &p_path) {
		// FIXME implement path splitting.

		auto &uvs = m_uvs[p_batch_index];

		if (uvs.empty()) {
			return std::optional<Value>();
		}

		const auto values = m_matrix.template values<1, 1>();

		const auto initial_uv = uvs.top();
		uvs.pop();

		Index u = std::get<0>(initial_uv);
		Index v = std::get<1>(initial_uv);
		const auto best_val = values(u, v)(p_batch_index);

		if (m_strategy.has_trace()) { // && m_path.wants_path()
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();
			p_path.begin(len_s, len_t);

			const auto traceback = m_matrix.template traceback<1, 1>();

			while (m_strategy.continue_traceback(u, v, values(u, v)(p_batch_index))) {
				const Index last_u = u;
				const Index last_v = v;

				const auto &t = traceback(u, v);
				u = t.u(p_batch_index, 0);
				v = t.v(p_batch_index, 0);

				p_path.step(last_u, last_v, u, v);
			}
		}

		return best_val;
	}
};


template<typename CellType, typename ProblemType, typename Strategy, typename Matrix>
using TracebackIterator2 = TracebackIterator<
	!traceback_type<CellType, ProblemType>::max_degree_1, CellType, ProblemType, Strategy, Matrix>;

template<typename Locality, typename Matrix>
inline TracebackIterator2<
	typename Locality::cell_type,
	typename Locality::problem_type,
	typename Locality::template TracebackStrategy<Matrix>,
	Matrix>
	make_traceback_iterator(
		const Locality &p_locality,
		Matrix &p_matrix) {

	return TracebackIterator2<
		typename Locality::cell_type,
		typename Locality::problem_type,
		typename Locality::template TracebackStrategy<Matrix>,
		Matrix>(

		typename Locality::template TracebackStrategy<Matrix>(p_locality, p_matrix),
		p_matrix
	);
}

template<typename Goal>
struct TracebackSupport {
};

template<>
struct TracebackSupport<goal::optimal_score> {
	template<typename CellType, typename ProblemType>
	using Accumulator = Accumulator<CellType, ProblemType>;

	static constexpr bool has_trace = false;
};

template<typename PathGoal>
struct TracebackSupport<goal::alignment<PathGoal>> {
	template<typename CellType, typename ProblemType>
	using Accumulator = TracingAccumulator<CellType, ProblemType>;

	static constexpr bool has_trace = true;
};

template<typename Direction, typename CellType>
class Optima {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename CellType::mask_vec_type MaskVec;

	constexpr static int BatchSize = CellType::batch_size;

private:
	const Value worst;
	ValueVec best_val;
	IndexVec best_i;
	IndexVec best_j;

public:
	inline Optima() : worst(Direction::template worst_val<Value>()) {
		best_val.fill(worst);
	}

	inline void add(const Index i, const Index j, const ValueVec &val) {
		const MaskVec mask = Direction::opt_q(val, best_val);
		best_val = Direction::opt(val, best_val);
		best_i = xt::where(mask, i, best_i);
		best_j = xt::where(mask, j, best_j);
	}

	template<typename Stack>
	inline void push(Stack &stack) {
		for (auto k : xt::flatnonzero<xt::layout_type::row_major>(Direction::opt_q(best_val, worst))) {
			stack[k].push(std::make_pair(best_i(k), best_j(k)));
		}
	}
};

struct LocalInitializers {
};

template<typename CellType, typename ProblemType>
class Local {
public:
	typedef LocalInitializers Initializers;

	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename CellType::mask_vec_type MaskVec;

	typedef CellType cell_type;
	typedef ProblemType problem_type;

	constexpr static bool is_global() {
		return false;
	}

	constexpr static Value ZERO = 0;
	constexpr static int BatchSize = CellType::batch_size;

public:
	typedef TracebackSupport<typename ProblemType::goal_type> TBS;
	typedef typename TBS::template Accumulator<CellType, ProblemType> Accumulator;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to(ValueCell &val, TracebackCell &tb) const {
		auto acc = Accumulator::create(val, tb);
		return acc.push(
			xt::zeros<Value>({BatchSize}),
			no_traceback<Index>(),
			no_traceback<Index>());
	}

	inline Local(const LocalInitializers &p_init) {
	}

	inline const char *name() const {
		return "local";
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		for (size_t i = 0; i < p_vector.size(); i++) {
			p_vector(i).fill(ZERO);
		}
	}

	inline Value zero() const {
		return ZERO;
	}

	template<typename Matrix, typename PathGoal>
	struct TracebackSeeds {
		typedef typename ProblemType::direction_type Direction;

		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto values = m_matrix.template values_n<1, 1>();

			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			Optima<Direction, CellType> optima;

			for (Index i = len_s - 1; i >= 0; i--) {
				for (Index j = len_t - 1; j >= 0; j--) {
					optima.add(i, j, values(i, j));
				}
			}

			optima.push(r_seeds);
		}
	};

	template<typename Matrix>
	struct TracebackSeeds<Matrix, goal::path::optimal::all> {
		typedef typename ProblemType::direction_type Direction;

		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto values = m_matrix.template values_n<1, 1>();

			ValueVec best_val = xt::zeros<Value>({BatchSize});

			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			for (Index i = len_s - 1; i >= 0; i--) {
				for (Index j = len_t - 1; j >= 0; j--) {
					best_val = Direction::opt(values(i, j), best_val);
				}
			}

			for (Index i = len_s - 1; i >= 0; i--) {
				for (Index j = len_t - 1; j >= 0; j--) {
					const ValueVec x = values(i, j);
					for (auto k : xt::flatnonzero<xt::layout_type::row_major>(xt::equal(x, best_val))) {
						if (Direction::is_opt(x(k), ZERO)) {
							r_seeds[k].push(std::make_pair(i, j));
						}
					}
				}
			}
		}
	};

	template<typename Matrix>
	class TracebackStrategy {
		typedef typename ProblemType::direction_type Direction;

		Matrix &m_matrix;

	public:
		inline TracebackStrategy(
			const Local<CellType, ProblemType> &p_locality,
			Matrix &p_matrix) :
			m_matrix(p_matrix) {
		}

		inline constexpr bool has_trace() const {
			return TBS::has_trace;
		}

		inline auto seeds() const {
			return TracebackSeeds<
				Matrix, typename ProblemType::goal_type::path_goal>{m_matrix};
		}

		inline bool continue_traceback(const Index u,
			const Index v,
			const Value val) const {

			return u >= 0 && v >= 0 && Direction::is_opt(val, Local::ZERO);
		}
	};
};

struct GlobalInitializers {
};

template<typename CellType, typename ProblemType>
class Global {
public:
	typedef GlobalInitializers Initializers;

	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	typedef CellType cell_type;
	typedef ProblemType problem_type;

	static constexpr bool is_global() {
		return true;
	}

	constexpr static int BatchSize = CellType::batch_size;

public:
	typedef TracebackSupport<typename ProblemType::goal_type> TBS;
	typedef typename TBS::template Accumulator<CellType, ProblemType> Accumulator;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to(ValueCell &val, TracebackCell &tb) const {
		return Accumulator::create(val, tb);
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		if (p_vector.size() != p_gap_cost.size()) {
			throw std::runtime_error(
				"size mismatch in Global::init_border_case");
		}

		for (size_t i = 0; i < p_vector.size(); i++) {
			p_vector(i).fill(p_gap_cost(i));
		}
	}

	inline Global(const GlobalInitializers&) {
	}

	template<typename Matrix, typename PathGoal>
	struct TracebackSeeds {
		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();
			for (int i = 0; i < BatchSize; i++) {
				r_seeds[i].push(std::make_pair(
					len_s - 1,
					len_t - 1));
			}
		}
	};

	template<typename Matrix>
	class TracebackStrategy {
		typedef typename ProblemType::direction_type Direction;

		Matrix &m_matrix;

	public:
		inline TracebackStrategy(
			const Global<CellType, ProblemType> &p_locality,
			Matrix &p_matrix) :
			m_matrix(p_matrix) {
		}

		inline constexpr bool has_trace() const {
			return TBS::has_trace;
		}

		inline auto seeds() const {
			return TracebackSeeds<
				Matrix, typename ProblemType::goal_type::path_goal>{m_matrix};
		}

		inline bool continue_traceback(
			const Index u,
			const Index v,
			const Value val) const {

			return u >= 0 && v >= 0;
		}
	};
};


struct SemiglobalInitializers {
};

template<typename CellType, typename ProblemType>
class Semiglobal {
public:
	typedef SemiglobalInitializers Initializers;

	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	typedef CellType cell_type;
	typedef ProblemType problem_type;

	static constexpr bool is_global() {
		return false;
	}

	constexpr static int BatchSize = CellType::batch_size;

public:
	typedef TracebackSupport<typename ProblemType::goal_type> TBS;
	typedef typename TBS::template Accumulator<CellType, ProblemType> Accumulator;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to(ValueCell &val, TracebackCell &tb) const {
		return Accumulator::create(val, tb);
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<Value, 1> &p_gap_cost) const {

		for (size_t i = 0; i < p_vector.size(); i++) {
			p_vector(i).fill(0);
		}
	}

	inline Semiglobal(const SemiglobalInitializers&) {
	}

	template<typename Matrix, typename PathGoal>
	struct TracebackSeeds {
		typedef typename ProblemType::direction_type Direction;

		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			const auto values = m_matrix.template values_n<1, 1>();

			const Index last_row = len_s - 1;
			const Index last_col = len_t - 1;

			Optima<Direction, CellType> optima;

			for (Index j = 0; j < len_t; j++) {
				optima.add(last_row, j, values(last_row, j));
			}

			for (Index i = 0; i < len_s; i++) {
				optima.add(i, last_col, values(i, last_col));
			}

			optima.push(r_seeds);
		}
	};

	template<typename Matrix>
	class TracebackStrategy {
		typedef typename ProblemType::direction_type Direction;

		Matrix &m_matrix;

	public:
		inline TracebackStrategy(
			const Semiglobal<CellType, ProblemType> &p_locality,
			Matrix &p_matrix) :
			m_matrix(p_matrix) {
		}

		inline constexpr bool has_trace() const {
			return TBS::has_trace;
		}

		inline auto seeds() const {
			return TracebackSeeds<
				Matrix, typename ProblemType::goal_type::path_goal>{m_matrix};
		}

		inline bool continue_traceback(
			const Index u,
			const Index v,
			const Value val) const {

			return u >= 0 && v >= 0;
		}
	};
};

template<typename CellType, typename ProblemType>
class Solution {
public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;
	typedef traceback_type<CellType, ProblemType> Traceback;

private:
	typedef std::pair<Index, Index> Coord;
	typedef std::pair<Coord, Coord> Edge;

public:
	xt::xtensor<Value, 3> m_values;
	xt::xtensor<typename Traceback::Single, 3> m_traceback;
	xt::xtensor<Index, 2> m_path;
	Value m_score;
	AlgorithmMetaDataRef m_algorithm;

	template<typename ValuesMatrix>
	void set_values(
		const ValuesMatrix &p_values,
		const int p_batch_index) {

		const size_t len_k = p_values.shape(0);
		const size_t len_s = p_values.shape(1);
		const size_t len_t = p_values.shape(2);

		xt::xtensor<Value, 3> values;
		m_values.resize({
			len_k, len_s, len_t
		});
		for (size_t k = 0; k < len_k; k++) {
			for (size_t i = 0; i < len_s; i++) {
				for (size_t j = 0; j < len_t; j++) {
					m_values(k, i, j) = p_values(k, i, j)(p_batch_index);
				}
			}
		}
	}

	template<typename TracebackMatrix>
	void set_traceback(
		const TracebackMatrix &p_traceback,
		const int p_batch_index) {

		const size_t len_k = p_traceback.shape(0);
		const size_t len_s = p_traceback.shape(1);
		const size_t len_t = p_traceback.shape(2);

		m_traceback.resize({
			len_k, len_s, len_t
		});
		for (size_t k = 0; k < len_k; k++) {
			for (size_t i = 0; i < len_s; i++) {
				for (size_t j = 0; j < len_t; j++) {
					m_traceback(k, i, j) = p_traceback(k, i, j).to_single(p_batch_index);
				}
			}
		}
	}

	inline void set_path(const xt::xtensor<Index, 2> &p_path) {
		m_path = p_path;
	}

	inline void set_score(const Value p_score) {
		m_score = p_score;
	}

	inline void set_algorithm(const AlgorithmMetaDataRef &p_algorithm) {
		m_algorithm = p_algorithm;
	}

	const xt::xtensor<Value, 3> &values() const {
		return m_values;
	}

	inline bool has_degree_1_traceback() const {
		return Traceback::max_degree_1;
	}

	const auto traceback_as_matrix() const {
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
					traceback(k, i, j, 0) = m_traceback(k, i, j).u(0);
					traceback(k, i, j, 1) = m_traceback(k, i, j).v(0);
				}
			}
		}
		return traceback;
	}

	const std::vector<xt::xtensor<Index, 3>> traceback_as_edges() const {
		const size_t len_k = m_traceback.shape(0);
		const size_t len_s = m_traceback.shape(1);
		const size_t len_t = m_traceback.shape(2);

		std::vector<xt::xtensor<Index, 3>> edges;
		edges.resize(len_k);

		const size_t avg_degree = 3; // an estimate
		std::vector<Edge> layer_edges;
		layer_edges.reserve(len_s * len_t * avg_degree);

		for (size_t k = 0; k < len_k; k++) {
			layer_edges.clear();
			for (size_t i = 0; i < len_s; i++) {
				for (size_t j = 0; j < len_t; j++) {
					const auto &cell = m_traceback(k, i, j);
					const size_t n_edges = cell.size();
					for (size_t q = 0; q < n_edges; q++) {
						layer_edges.emplace_back(std::make_pair<Coord, Coord>(
							std::make_pair<Index, Index>(
								static_cast<Index>(i) - 1,
								static_cast<Index>(j) - 1),
							std::make_pair<Index, Index>(
								cell.u(q),
								cell.v(q))));
					}
				}
			}

			xt::xtensor<Index, 3> &tensor = edges[k];
			const size_t n_edges = layer_edges.size();
			tensor.resize({
				n_edges, 2, 2
			});
			for (size_t i = 0; i < n_edges; i++) {
				const auto &edge = layer_edges[i];

				const auto &u = std::get<0>(edge);
				tensor(i, 0, 0) = std::get<0>(u);
				tensor(i, 0, 1) = std::get<1>(u);

				const auto &v = std::get<1>(edge);
				tensor(i, 1, 0) = std::get<0>(v);
				tensor(i, 1, 1) = std::get<1>(v);
			}
		}

		return edges;
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

template<typename CellType, typename ProblemType>
using SolutionRef = std::shared_ptr<Solution<CellType, ProblemType>>;


struct matrix_name {
	constexpr static int D = 0;
	constexpr static int P = 1;
	constexpr static int Q = 2;
};

struct shared_ptr_deref {
	template<typename T>
	inline T &operator()(const std::shared_ptr<T> &p) const {
		return *p.get();
	}
};

template<typename CellType, typename ProblemType, template<typename, typename> class Locality, int LayerCount>
class Solver {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename CellType::index_type Index;
	typedef typename ProblemType::direction_type Direction;

protected:
	const Locality<CellType, ProblemType> m_locality;
	MatrixFactory<CellType, ProblemType, LayerCount> m_factory;
	const AlgorithmMetaDataRef m_algorithm;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to_nolocal(ValueCell &val, TracebackCell &tb) const {
		typedef TracebackSupport<typename ProblemType::goal_type> TBS;
		typedef typename TBS::template Accumulator<CellType, ProblemType> Accumulator;

		return Accumulator::create(val, tb);
	}

public:
	inline Solver(
		const typename Locality<CellType, ProblemType>::Initializers &p_locality_init,
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

	inline constexpr int batch_size() const {
		return CellType::batch_size;
	}

	template<int Layer>
	inline auto matrix(const Index len_s, const Index len_t) {
		return m_factory.make<Layer>(len_s, len_t);
	}

	inline ValueVec score(
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = m_factory.template make<matrix_name::D>(len_s, len_t);
		auto tb = make_traceback_iterator(m_locality, matrix);

		ValueVec scores;
		build_nothing nothing;
		for (int i = 0; i < CellType::batch_size; i++) {
			const auto tb_val = tb.next(i, nothing);
			scores(i) = tb_val.value_or(Direction::template worst_val<Value>());
		}
		return scores;
	}

	template<typename AlignmentRef, typename Deref=shared_ptr_deref>
	inline void alignment(
		const size_t len_s,
		const size_t len_t,
		const std::array<AlignmentRef, CellType::batch_size> &alignments) const {

		auto matrix = m_factory.template make<matrix_name::D>(len_s, len_t);
		auto tb = make_traceback_iterator(m_locality, matrix);
		const auto deref = Deref();

		for (int i = 0; i < CellType::batch_size; i++) {
			auto &alignment = deref(alignments[i]);
			build_alignment<decltype(alignment)> build(alignment);
			const auto tb_val = tb.next(i, build);
			alignment.set_score(tb_val.value_or(
				Direction::template worst_val<Value>()));
		}
	}

	template<typename AlignmentRef, typename SolutionRef, typename Deref=shared_ptr_deref>
	void solution(
		const size_t len_s,
		const size_t len_t,
		const std::array<AlignmentRef, CellType::batch_size> &alignments,
		const std::array<SolutionRef, CellType::batch_size> &solutions) const {

		auto matrix = m_factory.template make<matrix_name::D>(len_s, len_t);
		auto tb = make_traceback_iterator(m_locality, matrix);
		const auto deref = Deref();

		for (int i = 0; i < CellType::batch_size; i++) {
			auto &alignment = deref(alignments[i]);
			auto &solution = deref(solutions[i]);

			auto build = build_multiple<build_path<Index>, build_alignment<decltype(alignment)>>(
				build_path<Index>(), build_alignment<decltype(alignment)>(alignment)
			);

			solution.set_values(m_factory.all_layers().values(len_s, len_t), i);
			solution.set_traceback(m_factory.all_layers().traceback(len_s, len_t), i);

			const auto tb_val = tb.next(i, build);
			solution.set_path(build.template get<0>().path());

			const auto score = tb_val.value_or(Direction::template worst_val<Value>());
			solution.set_score(score);
			alignment.set_score(score);

			solution.set_algorithm(m_algorithm);
		}
	}
};

template<typename CellType, typename ProblemType, template<typename, typename> class Locality, int LayerCount>
using AlignmentSolver = Solver<CellType, ProblemType, Locality, LayerCount>;

template<typename CellType, typename ProblemType, template<typename, typename> class Locality>
class LinearGapCostSolver final : public AlignmentSolver<CellType, ProblemType, Locality, 1> {
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
	typedef typename ProblemType::goal_type Goal;
	typedef typename ProblemType::direction_type Direction;
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename Locality<CellType, ProblemType>::Initializers LocalityInit;

private:
	const Value m_gap_cost_s;
	const Value m_gap_cost_t;

public:
	typedef Value GapCostSpec;

	inline LinearGapCostSolver(
		const Value p_gap_cost_s,
		const Value p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const LocalityInit p_locality_init = LocalityInit()) :

		AlignmentSolver<CellType, ProblemType, Locality, 1>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>(
				Locality<CellType, ProblemType>::is_global() ?
					"Needleman-Wunsch": "Smith-Waterman",
				"n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto values = this->m_factory.template values<matrix_name::D>();
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

	template<typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = this->m_factory.template make<matrix_name::D>(len_s, len_t);

		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

				this->m_locality.accumulate_to(
					values(u, v), traceback(u, v))
				.push(
					values(u - 1, v - 1) + pairwise(u, v),
					u - 1, v - 1)
				.push(
					values(u - 1, v) + this->m_gap_cost_s * gap_sgn,
					u - 1, v)
				.push(
					values(u, v - 1) + this->m_gap_cost_t * gap_sgn,
					u, v - 1)
				.done();
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

template<typename CellType, typename ProblemType, template<typename, typename> class Locality>
class AffineGapCostSolver final : public AlignmentSolver<CellType, ProblemType, Locality, 3> {
public:
	// Gotoh, O. (1982). An improved algorithm for matching biological sequences.
	// Journal of Molecular Biology, 162(3), 705–708. https://doi.org/10.1016/0022-2836(82)90398-9

	typedef typename ProblemType::goal_type Goal;
	typedef typename ProblemType::direction_type Direction;
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename Locality<CellType, ProblemType>::Initializers LocalityInit;

	typedef AffineCost<Value> Cost;

private:
	const Cost m_gap_cost_s;
	const Cost m_gap_cost_t;

public:
	inline AffineGapCostSolver(
		const Cost &p_gap_cost_s,
		const Cost &p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const LocalityInit p_locality_init = LocalityInit()) :

		AlignmentSolver<CellType, ProblemType, Locality, 3>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>("Gotoh", "n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto matrix_D = this->m_factory.template make<matrix_name::D>(p_max_len_s, p_max_len_t);
		auto matrix_P = this->m_factory.template make<matrix_name::P>(p_max_len_s, p_max_len_t);
		auto matrix_Q = this->m_factory.template make<matrix_name::Q>(p_max_len_s, p_max_len_t);

		auto D = matrix_D.template values<0, 0>();
		auto P = matrix_P.template values<0, 0>();
		auto Q = matrix_Q.template values<0, 0>();

		const auto inf = std::numeric_limits<Value>::infinity() * (Direction::is_minimize() ? 1 : -1);

		for (auto &cell : xt::view(Q, xt::all(), 0)) {
			cell.fill(inf);
		}
		for (auto &cell : xt::view(P, 0, xt::all())) {
			cell.fill(inf);
		}

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
			e.clear();
		}
		for (auto &e : xt::view(tb_P, xt::all(), 0)) {
			e.clear();
		}
		for (auto &e : xt::view(tb_Q, 0, xt::all())) {
			e.clear();
		}
		for (auto &e : xt::view(tb_Q, xt::all(), 0)) {
			e.clear();
		}
	}

	template<typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix_D = this->m_factory.template make<matrix_name::D>(len_s, len_t);
		auto matrix_P = this->m_factory.template make<matrix_name::P>(len_s, len_t);
		auto matrix_Q = this->m_factory.template make<matrix_name::Q>(len_s, len_t);

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
					this->accumulate_to_nolocal(
						P(i, j), tb_P(i, j))
					.push(
						D(i - 1, j) + m_gap_cost_s.w1() * gap_sgn,
						i - 1, j)
					.push(
						P(i - 1, j) + m_gap_cost_s.u * gap_sgn,
						tb_P(i - 1, j))
					.done();
				}

				// Gotoh formula (5)
				{
					this->accumulate_to_nolocal(
						Q(i, j), tb_Q(i, j))
					.push(
						D(i, j - 1) + m_gap_cost_t.w1() * gap_sgn,
						i, j - 1)
					.push(
						Q(i, j - 1) + m_gap_cost_t.u * gap_sgn,
						tb_Q(i, j - 1))
					.done();
				}

				// Gotoh formula (1)
				{
					this->m_locality.accumulate_to(
						D(i, j), tb_D(i, j))
					.push(
						D(i - 1, j - 1) + pairwise(i, j),
						i - 1, j - 1)
					.push(
						P(i, j),
						tb_P(i, j))
					.push(
						Q(i, j),
						tb_Q(i, j))
					.done();
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

template<typename CellType, typename ProblemType, template<typename, typename> class Locality>
class GeneralGapCostSolver final : public AlignmentSolver<CellType, ProblemType, Locality, 1> {
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
	typedef typename ProblemType::goal_type Goal;
	typedef typename ProblemType::direction_type Direction;
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;
	typedef typename Locality<CellType, ProblemType>::Initializers LocalityInit;

private:
	const xt::xtensor<Value, 1> m_gap_cost_s;
	const xt::xtensor<Value, 1> m_gap_cost_t;

public:
	typedef GapTensorFactory<Value> GapCostSpec;

	inline GeneralGapCostSolver(
		const GapTensorFactory<Value> &p_gap_cost_s,
		const GapTensorFactory<Value> &p_gap_cost_t,
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const LocalityInit p_locality_init = LocalityInit()) :

		AlignmentSolver<CellType, ProblemType, Locality, 1>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>("Waterman-Smith-Beyer", "n^3", "n^2")),
		m_gap_cost_s(p_gap_cost_s(p_max_len_s + 1)),
		m_gap_cost_t(p_gap_cost_t(p_max_len_t + 1)) {

		check_gap_tensor_shape(m_gap_cost_s, p_max_len_s + 1);
		check_gap_tensor_shape(m_gap_cost_t, p_max_len_t + 1);

		auto values = this->m_factory.template values<matrix_name::D>();
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

	template<typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = this->m_factory.template make<matrix_name::D>(len_s, len_t);

		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();
		constexpr Value gap_sgn = Direction::is_minimize() ? 1 : -1;

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

				this->m_locality.accumulate_to(
					values(u, v), traceback(u, v))

				.push(
					values(u - 1, v - 1) + pairwise(u, v),
					u - 1, v - 1)

				.push_many([this, u, v, gap_sgn, &values] (auto acc) {
					for (Index k = -1; k < u; k++) {
						acc.push(
							values(k, v) + this->m_gap_cost_s(u - k) * gap_sgn,
							k, v);
					}
				})

				.push_many([this, u, v, gap_sgn, &values] (auto acc) {
					for (Index k = -1; k < v; k++) {
						acc.push(
							values(u, k) + this->m_gap_cost_t(v - k) * gap_sgn,
							u, k);
					}
				})

				.done();
			}
		}
	}
};

template<typename CellType, typename ProblemType>
class DynamicTimeSolver final : public Solver<CellType, ProblemType, Global, 1> {
public:
	typedef typename ProblemType::goal_type Goal;
	typedef typename ProblemType::direction_type Direction;
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_type Value;

	inline DynamicTimeSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		Solver<CellType, ProblemType, Global, 1>(
			GlobalInitializers(),
			p_max_len_s,
			p_max_len_t,
			std::make_shared<AlgorithmMetaData>("DTW", "n^2", "n^2")) {

		auto values = this->m_factory.template values<matrix_name::D>();
		const auto worst = std::numeric_limits<Value>::infinity() * (Direction::is_minimize() ? 1 : -1);
		for (auto &cell : values) {
			cell.fill(worst);
		}
		values.at(0, 0).fill(0);
	}

	template<typename Pairwise>
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

		auto matrix = this->m_factory.template make<matrix_name::D>(len_s, len_t);
		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();

		for (Index u = 0; static_cast<size_t>(u) < len_s; u++) {
			for (Index v = 0; static_cast<size_t>(v) < len_t; v++) {

				this->m_locality.accumulate_to(
					values(u, v), traceback(u, v))
				.push(
					values(u - 1, v - 1),
					u - 1, v - 1)
				.push(
					values(u - 1, v),
					u - 1, v)
				.push(
					values(u, v - 1),
					u, v - 1)
				.add(pairwise(u, v))
				.done();
			}
		}
	}
};

} // namespace pyalign

#endif // __PYALIGN_SOLVER__
