#ifndef __PYALIGN_SOLVER_H__
#define __PYALIGN_SOLVER_H__ 1

#include "pyalign/algorithm/common.h"
#include <stack>
#include <unordered_set>

namespace pyalign {
namespace core {

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

	static std::string to_text(const char *p_name, const size_t p_len, const size_t p_max) {
		std::stringstream err;
		err << "requested maximum length " << p_len <<
			" for " << p_name <<
			" exceeds maximum supported sequence length in this implementation " << p_max;
		return err.str();
	}

public:
	exceeded_implementation_length(const char *p_name, const size_t p_len, const size_t p_max) :
		exceeded_length(p_len, p_max), m_err(to_text(p_name, p_len, p_max)) {
	}

	virtual char const *what() const noexcept {
		return m_err.c_str();
	}
};

class exceeded_configured_length : public exceeded_length {
	const std::string m_err;

	static std::string to_text(const char *p_name, const size_t p_len, const size_t p_max) {
		std::stringstream err;
		err << "sequence " << p_name << " of length " << p_len <<
			" exceeds configured maximum length " << p_max;
		return err.str();
	}

public:
	exceeded_configured_length(const char *p_name, const size_t p_len, const size_t p_max) :
		exceeded_length(p_len, p_max), m_err(to_text(p_name, p_len, p_max)) {
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
	typedef typename CellType::index_type index_type;
	typedef typename CellType::index_vec_type index_vec_type;
	index_vec_type v;
	v.fill(std::numeric_limits<index_type>::min());
	return v;
}

template<typename CellType>
struct traceback_1 {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename CellType::mask_vec_type mask_vec_type;

private:
	struct {
		index_vec_type u;
		index_vec_type v;
	} uv;

public:
	static constexpr bool max_degree_1 = true;

	struct Single {
		index_type _u;
		index_type _v;

		inline index_type size() const {
			return 1;
		}

		inline index_type u(const size_t i) const {
			return _u;
		}

		inline index_type v(const size_t i) const {
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

	inline void init(const index_type p_u, const index_type p_v) {
		uv.u.fill(p_u);
		uv.v.fill(p_v);
	}

	inline void init(const index_type p_u, const index_type p_v, const mask_vec_type &mask) {
		const index_vec_type u = xt::full_like(index_vec_type(), p_u);
		const index_vec_type v = xt::full_like(index_vec_type(), p_v);
		uv.u = xt::where(mask, u, uv.u);
		uv.v = xt::where(mask, v, uv.v);
	}

	inline void init(const traceback_1 &tb, const mask_vec_type &mask) {
		uv.u = xt::where(mask, tb.uv.u, uv.u);
		uv.v = xt::where(mask, tb.uv.v, uv.v);
	}

	inline void push(const index_type p_u, const index_type p_v, const mask_vec_type &mask) {
		const index_vec_type u = xt::full_like(index_vec_type(), p_u);
		const index_vec_type v = xt::full_like(index_vec_type(), p_v);
		uv.u = xt::where(mask, u, uv.u);
		uv.v = xt::where(mask, v, uv.v);
	}

	inline void push(const traceback_1 &tb, const mask_vec_type &mask) {
		uv.u = xt::where(mask, tb.uv.u, uv.u);
		uv.v = xt::where(mask, tb.uv.v, uv.v);
	}

	inline index_type u(const int batch_index, const size_t i) const {
		return uv.u(batch_index);
	}

	inline index_type v(const int batch_index, const size_t i) const {
		return uv.v(batch_index);
	}

	inline index_type size(const int batch_index) const {
		return uv.u(batch_index) != no_traceback<index_type>() ? 1 : 0;
	}
};

template<typename CellType>
struct traceback_n {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename CellType::mask_vec_type mask_vec_type;

	static constexpr int BatchSize = CellType::batch_size;

	struct Pt {
		index_type u;
		index_type v;
	};

	std::vector<Pt> pts[BatchSize];

public:
	static constexpr bool max_degree_1 = false;

	struct Single {
		std::vector<Pt> pts;

		inline index_type size() const {
			return pts.size();
		}

		inline index_type u(const size_t i) const {
			return i < pts.size() ? pts[i].u : no_traceback<index_type>();
		}

		inline index_type v(const size_t i) const {
			return i < pts.size() ? pts[i].v : no_traceback<index_type>();
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

	inline void init(const index_type p_u, const index_type p_v) {
		for (int i = 0; i < BatchSize; i++) {
			pts[i].clear();
			pts[i].emplace_back(Pt{p_u, p_v});
		}
	}

	inline void init(const index_type p_u, const index_type p_v, const mask_vec_type &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			pts[i].clear();
			pts[i].emplace_back(Pt{p_u, p_v});
		}
	}

	inline void init(const traceback_n &tb, const mask_vec_type &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			pts[i] = tb.pts[i];
		}
	}

	inline void push(const index_type p_u, const index_type p_v, const mask_vec_type &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			pts[i].emplace_back(Pt{p_u, p_v});
		}
	}

	inline void push(const traceback_n &tb, const mask_vec_type &mask) {
		for (auto i : xt::flatnonzero<xt::layout_type::row_major>(mask)) {
			for (const auto &pt : tb.pts[i]) {
				pts[i].push_back(pt);
			}
		}
	}

	inline index_type u(const int batch_index, const size_t i) const {
		return i < pts[batch_index].size() ?
			pts[batch_index][i].u :
			no_traceback<index_type>();
	}

	inline index_type v(const int batch_index, const size_t i) const {
		return i < pts[batch_index].size() ?
			pts[batch_index][i].v :
			no_traceback<index_type>();
	}

	inline index_type size(const int batch_index) const {
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
using _traceback_type = typename traceback_cell_type_factory<
	typename ProblemType::goal_type::path_goal,
	CellType>::traceback_cell_type;


template<typename CellType, typename ProblemType>
class Matrix;

template<typename CellType, typename ProblemType>
class MatrixFactory;

template<typename CellType, typename ProblemType>
using MatrixFactoryRef = std::shared_ptr<MatrixFactory<CellType, ProblemType>>;

template<typename CellType, typename ProblemType>
class MatrixFactory {
public:
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef _traceback_type<CellType, ProblemType> traceback_type;

protected:
	friend class Matrix<CellType, ProblemType>;

	struct Data {
		xt::xtensor<value_vec_type, 3> values;
		xt::xtensor<traceback_type, 3> traceback;
	};

	const std::unique_ptr<Data> m_data;
	const size_t m_max_len_s;
	const size_t m_max_len_t;
	const uint16_t m_layer_count;

	inline void check_size_against_max(
		const char *p_name,
		const size_t p_len,
		const size_t p_max) const {
		if (p_len > p_max) {
			throw exceeded_configured_length(p_name, p_len, p_max);
		}
	}

	inline void check_size_against_implementation_limit(
		const char *p_name,
		const size_t p_len) const {
		const size_t max = size_t(std::numeric_limits<index_type>::max()) >> 1;
		if (p_len > max) {
			throw exceeded_implementation_length(p_name, p_len, max);
		}
	}

public:
	inline MatrixFactory(
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const uint16_t p_layer_count) :

		m_data(std::make_unique<Data>()),
		m_max_len_s(p_max_len_s),
		m_max_len_t(p_max_len_t),
		m_layer_count(p_layer_count) {

		check_size_against_implementation_limit("s", p_max_len_s);
		check_size_against_implementation_limit("t", p_max_len_t);

		m_data->values.resize({
			p_layer_count,
			m_max_len_s + 1,
			m_max_len_t + 1
		});
		m_data->traceback.resize({
			p_layer_count,
			m_max_len_s + 1,
			m_max_len_t + 1
		});

		for (int k = 0; k < p_layer_count; k++) {
			for (size_t i = 0; i < m_max_len_s + 1; i++) {
				m_data->traceback(k, i, 0).clear();
			}
			for (size_t j = 0; j < m_max_len_t + 1; j++) {
				m_data->traceback(k, 0, j).clear();
			}
		}
	}

	template<int Layer>
	inline Matrix<CellType, ProblemType> make(
		const index_type len_s, const index_type len_t) const;

	template<int Layer>
	inline MatrixFactoryRef<CellType, ProblemType> copy(
		const index_type len_s, const index_type len_t) const {

	    auto ref = std::make_shared<MatrixFactory<CellType, ProblemType>>(
	        len_s, len_t, 1);

        xt::view(ref->m_data->values, 0, xt::all(), xt::all()) = xt::view(
            m_data->values, Layer, xt::range(0, len_s + 1), xt::range(0, len_t + 1));
        xt::view(ref->m_data->traceback, 0, xt::all(), xt::all()) = xt::view(
            m_data->traceback, Layer, xt::range(0, len_s + 1), xt::range(0, len_t + 1));

	    return ref;
    }

	inline index_type max_len_s() const {
		return m_max_len_s;
	}

	inline index_type max_len_t() const {
		return m_max_len_t;
	}

	template<int Layer>
	inline auto values() const {
		return xt::view(m_data->values, Layer, xt::all(), xt::all());
	}

	struct all_layers_accessor {
		Data &m_data;

		inline auto values(const index_type len_s, const index_type len_t) const {
			return xt::view(
				m_data.values,
				xt::all(), xt::range(0, len_s + 1), xt::range(0, len_t + 1));
		}

		inline auto traceback(const index_type len_s, const index_type len_t) const {
			return xt::view(
				m_data.traceback,
				xt::all(), xt::range(0, len_s + 1), xt::range(0, len_t + 1));
		}
	};

	all_layers_accessor all_layers() const {
		return all_layers_accessor{*m_data.get()};
	}

	template<typename Solution>
	void copy_solution_data(
		const size_t len_s,
		const size_t len_t,
		const int i,
		Solution &solution) {

		solution.set_values(this->all_layers().values(len_s, len_t), i);
		solution.set_traceback(this->all_layers().traceback(len_s, len_t), i);
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

template<typename CellType, typename ProblemType>
class Matrix {
public:
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_type index_type;

private:
	const MatrixFactory<CellType, ProblemType> &m_factory;
	const index_type m_len_s;
	const index_type m_len_t;
	const uint16_t m_layer;

public:
	inline Matrix(
		const MatrixFactory<CellType, ProblemType> &factory,
		const index_type len_s,
		const index_type len_t,
		const uint16_t layer) :

	    m_factory(factory),
	    m_len_s(len_s),
	    m_len_t(len_t),
	    m_layer(layer) {
	}

	inline index_type len_s() const {
		return m_len_s;
	}

	inline index_type len_t() const {
		return m_len_t;
	}

	template<int i0, int j0>
	inline auto values_n() const {
		return shift_xview<i0, j0>(xt::view(
			m_factory.m_data->values, m_layer, xt::all(), xt::all()));
	}

	template<int i0, int j0>
	inline auto traceback_n() const {
		return shift_xview<i0, j0>(xt::view(
			m_factory.m_data->traceback, m_layer, xt::all(), xt::all()));
	}

	struct assume_non_negative_indices {
	};

	template<int i0, int j0>
	inline auto values() const {
		return xt::view(
			m_factory.m_data->values,
			m_layer,
			xt::range(i0, m_len_s + 1),
			xt::range(j0, m_len_t + 1));
	}

	template<int i0, int j0>
	inline auto traceback() const {
		return xt::view(
			m_factory.m_data->traceback,
			m_layer,
			xt::range(i0, m_len_s + 1),
			xt::range(j0, m_len_t + 1));
	}

	void print_values(int batch_index = 0) const {
		auto m = this->values<1, 1>();
		for (index_type i = -1; i < m_len_s; i++) {
			for (index_type j = -1; j < m_len_t; j++) {
				const auto &cell = m(i, j);
				std::cout << "(" << i << "," << j << "): " << cell(batch_index) << std::endl;
			}
		}
	}

	void print_traceback(int batch_index = 0) const {
		auto m = this->traceback<1, 1>();
		for (index_type i = -1; i < m_len_s; i++) {
			for (index_type j = -1; j < m_len_t; j++) {
				std::cout << "(" << i << "," << j << "): ";
				const auto &cell = m(i, j);
				for (int k = 0; k < cell.size(batch_index); k++) {
					if (k > 0) {
						std::cout << ", ";
					}
					std::cout << "(" << cell.u(batch_index, k) << ", " << cell.v(batch_index, k) << ")";
				}
				std::cout << std::endl;
			}
		}
	}
};

template<typename CellType, typename ProblemType>
template<int Layer>
inline Matrix<CellType, ProblemType>
MatrixFactory<CellType, ProblemType>::make(
	const index_type len_s, const index_type len_t) const {

	if (Layer >= m_layer_count) {
		throw std::invalid_argument("layer index exceeds layer count");
	}
	check_size_against_max("s", len_s, m_max_len_s);
	check_size_against_max("t", len_t, m_max_len_t);
	return Matrix<CellType, ProblemType>(*this, len_s, len_t, Layer);
}

template<typename CellType>
struct CompressedPath {
	typedef typename CellType::index_type index_type;

	typedef xt::xtensor_fixed<index_type, xt::xshape<2>> Coord;

	std::vector<Coord> path;

	inline CompressedPath() {
	}

	inline bool operator==(const CompressedPath &p) const {
	    return path == p.path;
	}
};

template<typename CellType>
struct CompressedPathHash {
	typedef typename CellType::index_type index_type;

    size_t operator()(const CompressedPath<CellType>& v) const {
        std::hash<index_type> hasher;
        size_t seed = 0;
        for (const auto &p : v.path) {
            for (int i = 0; i < 2; i++) {
                seed ^= hasher(p(i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }
};

template<typename CellType, typename ProblemType>
class build_val {
public:
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_type index_type;
	typedef typename ProblemType::direction_type direction_type;

private:
	value_type m_val;

public:
	inline build_val() : m_val(direction_type::template worst_val<value_type>()) {
	}

	inline value_type val() const {
		return m_val;
	}

	inline void begin(
		const index_type len_s,
		const index_type len_t) {
	}

	inline void step(
		const index_type u,
		const index_type v) {
	}

    template<typename F>
    inline bool check_emit(const F &f) const {
        return true;
    }

	inline void emit(
		const value_type p_val) {
		m_val = p_val;
	}

	inline size_t size() const {
		return 0;
	}

	inline void go_back(
		const size_t p_size) {
	}
};

template<typename CellType>
class path_compressor {
	typedef typename CellType::index_type index_type;

    index_type m_last_u;
    index_type m_last_v;
    bool m_empty;

public:
    inline path_compressor() {
        m_empty = true;
    }

    inline bool empty() const {
        return m_empty;
    }

    inline void begin() {
        m_empty = true;
    }

    template<typename F>
    inline void step(
        const index_type u,
        const index_type v,
        const F &f) {

        if (m_empty) {
            m_empty = false;
        } else if ((u != m_last_u && v != m_last_v)) { //} || u < 0 || v < 0) {
            if (m_last_u >= 0 && m_last_v >= 0) {
                f(m_last_u, m_last_v);
            }
        }

        m_last_u = u;
        m_last_v = v;
    }
};

template<typename CellType>
class make_path_compressor {
public:
    const path_compressor<CellType> make() const {
        return path_compressor<CellType>();
    }
};


template<typename CellType, typename ProblemType>
class build_path {
public:
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_type index_type;
	typedef typename ProblemType::direction_type direction_type;

private:
	typedef xt::xtensor_fixed<index_type, xt::xshape<2>> Coord;

	std::vector<Coord> m_path;
	value_type m_val;

public:
	inline build_path() : m_val(direction_type::template worst_val<value_type>()) {
	}

	inline value_type val() const {
		return m_val;
	}

	inline void begin(
		const index_type len_s,
		const index_type len_t) {

		m_path.reserve(len_s + len_t);
		m_path.clear();
		m_val = direction_type::template worst_val<value_type>();
	}

	inline void step(
		const index_type u,
		const index_type v) {

        m_path.push_back(Coord{u, v});
	}

    template<typename F>
    inline bool check_emit(const F &f) const {
        return true;
    }

	template<typename Value>
	inline void emit(Value val) {
		m_val = val;
	}

	inline xt::xtensor<index_type, 2> path() const {
		xt::xtensor<index_type, 2> path;
		path.resize({m_path.size(), 2});
		for (size_t i = 0; i < m_path.size(); i++) {
			xt::view(path, i, xt::all()) = m_path[i];
		}
		return path;
	}

	inline size_t size() const {
		return m_path.size();
	}

	inline void go_back(
		const size_t p_size) {
		m_path.resize(p_size);
	}

	template<typename F>
	inline void iterate(
	    const make_path_compressor<CellType> &filter_factory,
	    const F &f) const {

		auto filter = filter_factory.make();
		const size_t n = m_path.size();
		for (size_t i = 0; i < n; i++) {
			filter.step(
				m_path[i](0), m_path[i](1), [&f] (auto u, auto v) {
				    f(u, v);
				}
			);
		}
	}
};

template<typename CellType, typename ProblemType>
struct build_alignment {
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_type index_type;
	typedef typename ProblemType::direction_type direction_type;

	template<typename Alignment>
	struct unbuffered {
		Alignment &m_alignment;
		make_path_compressor<CellType> m_filter_factory;
		index_type m_steps;
		path_compressor<CellType> m_filter;

	public:
		inline unbuffered(
		    Alignment &p_alignment,
		    const make_path_compressor<CellType> &p_filter_factory) :

			m_alignment(p_alignment),
			m_filter_factory(p_filter_factory),
			m_steps(0) {
		}

		inline void begin(
			const index_type len_s,
			const index_type len_t) {

			if (m_steps > 0) {
                throw std::runtime_error(
                    "internal error: called begin() on non-empty unbuffered alignment builder");
			}

			m_alignment.resize(len_s, len_t);
			m_filter = m_filter_factory.make();
			m_steps = 0;
		}

		inline void step(
			const index_type u,
			const index_type v) {

            m_filter.step(u, v, [this] (auto u, auto v) {
                m_alignment.add_edge(u, v);
            });

            m_steps += 1;
		}

        template<typename F>
		inline bool check_emit(const F &f) const {
            return true;
		}

		inline void emit(value_type val) {
			m_alignment.set_score(val);
		}

		inline size_t size() const {
			return m_steps;
		}

		inline void go_back(
			const size_t p_size) {

            throw std::runtime_error(
                "internal error: called go_back() on unbuffered alignment builder");
        }
    };

	template<typename Alignment>
	class buffered {
		build_path<CellType, ProblemType> m_path;
		mutable CompressedPath<CellType> m_compressed_path;
		const make_path_compressor<CellType> m_filter_factory;
		index_type m_len_s;
		index_type m_len_t;

		inline void update_compressed_path() const {
		    m_compressed_path.path.reserve(m_len_s + m_len_t);
		    m_compressed_path.path.clear();
			m_path.iterate(m_filter_factory, [this] (
				const index_type u,
				const index_type v) {

				typedef typename CompressedPath<CellType>::Coord Coord;

                m_compressed_path.path.push_back(Coord{u, v});
			});
		}
	public:
		inline buffered(const make_path_compressor<CellType> &p_filter_factory) :
		    m_filter_factory(p_filter_factory),
		    m_len_s(0),
		    m_len_t(0) {
		}

		inline void begin(
			const index_type len_s,
			const index_type len_t) {

			m_len_s = len_s;
			m_len_t = len_t;
			m_path.begin(len_s, len_t);
		}

		inline void step(
			const index_type u,
			const index_type v) {

			m_path.step(u, v);
		}

        template<typename F>
		inline bool check_emit(const F &f) const {
    		update_compressed_path();
		    return f(m_compressed_path);
		}

		inline void emit(value_type val) {
            m_path.emit(val);
		}

		inline size_t size() const {
			return m_path.size();
		}

		inline void go_back(
			const size_t p_size) {
			m_path.go_back(p_size);
		}

		inline void copy_to(
			Alignment &p_alignment) const {

			p_alignment.resize(m_len_s, m_len_t);

			m_path.iterate(m_filter_factory, [&p_alignment] (
				const index_type u,
				const index_type v) {

                p_alignment.add_edge(u, v);
			});

			p_alignment.set_score(m_path.val());
		}
	};
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
		const Index u,
		const Index v) {
	}

    template<typename F>
    inline bool check_emit(const F &f) const {
        return true;
    }

	template<typename Value>
	inline void emit(
		const Value p_val) {
	}

	inline size_t size() {
		return 0;
	}

	inline void check_size(
		const size_t p_size) const {
	}

	inline void go_back(
		const size_t p_size) {
	}

	template<int i>
	const auto &get() const {
		throw std::invalid_argument(
			"illegal index in build_multiple");
	}
};

template<typename BuildHead, typename... BuildRest>
class build_multiple<BuildHead, BuildRest...>;

template<int i, typename BuildHead, typename... BuildRest>
struct getter {
	const BuildHead &head;
	const build_multiple<BuildRest...> &rest;

	const auto &get() const {
		return rest.template get<i-1>();
	}
};

template<typename BuildHead, typename... BuildRest>
struct getter<0, BuildHead, BuildRest...> {
	const BuildHead &head;
	const build_multiple<BuildRest...> &rest;

	const auto &get() const {
		return head;
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
		const Index u,
		const Index v) {

		m_head.step(u, v);
		m_rest.step(u, v);
	}

    template<typename F>
    inline bool check_emit(const F &f) const {
        return m_head.check_emit(f) && m_rest.check_emit(f);
    }

	template<typename Value>
	inline void emit(
		const Value p_val) {

		m_head.emit(p_val);
		m_rest.emit(p_val);
	}

	inline size_t size() {
		const size_t s = m_head.size();
		m_rest.check_size(s);
		return s;
	}

	inline void check_size(
		const size_t p_size) const {
		if (p_size != m_head.size()) {
			throw std::runtime_error(
				"inconsistent size in build_multiple");
		}
		m_rest.check_size(p_size);
	}

	inline void go_back(
		const size_t p_size) {
		m_head.go_back(p_size);
		m_rest.go_back(p_size);
	}

	template<int i>
	const auto &get() const {
		return getter<i, BuildHead, BuildRest...>{m_head, m_rest}.get();
	}
};

template<typename CellType, typename ProblemType>
class Accumulator {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef _traceback_type<CellType, ProblemType> traceback_type;

public:
	struct cont {
		value_vec_type &m_val;

		inline auto push(
			const value_vec_type &val,
			const index_type u,
			const index_type v) {

			m_val = direction_type::opt(val, m_val);
			return cont{m_val};
		}

		inline auto push(
			const value_vec_type &val,
			const traceback_type &tb) {

			m_val = direction_type::opt(val, m_val);
			return cont{m_val};
		}

		template<typename F>
		inline auto push_many(const F &f) {
			f(cont{m_val});
			return cont{m_val};
		}

		inline auto add(const value_vec_type &val) {
			m_val += val;
			return cont{m_val};
		}

		inline void done() const {
		}
	};

	struct init {
	protected:
		friend class Accumulator;

		inline explicit init(value_vec_type &p_val) : m_val(p_val) {
		}

		init(const init&) = delete;
		init& operator=(init const&) = delete;

	public:
		value_vec_type &m_val;

		inline auto push(
			const value_vec_type &val,
			const index_type u,
			const index_type v) {

			m_val = val;
			return cont{m_val};
		}

		inline auto push(
			const value_vec_type &val,
			const traceback_type &tb) {

			m_val = val;
			return cont{m_val};
		}
	};

	static inline auto create(
		value_vec_type &p_val,
		traceback_type &p_tb) {

		return init{p_val};
	}
};

template<typename CellType, typename ProblemType>
class TracingAccumulator {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef typename CellType::mask_vec_type mask_vec_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef _traceback_type<CellType, ProblemType> traceback_type;

	static constexpr int BatchSize = CellType::batch_size;

	struct cont {
		value_vec_type &m_val;
		traceback_type &m_tb;

		inline auto push(
			const value_vec_type &val,
			const index_type u,
			const index_type v) {

			if (BatchSize == 1 && traceback_type::max_degree_1) {
				if (direction_type::is_opt(val(0), m_val(0))) {
					m_val = val;
					m_tb.init(u, v);
				}
			} else {
				m_tb.init(u, v, direction_type::opt_q(val, m_val));
				if (!traceback_type::max_degree_1) {
					m_tb.push(u, v, xt::equal(val, m_val));
				}

				m_val = direction_type::opt(val, m_val);
			}

			return cont{m_val, m_tb};
		}

		inline auto push(
			const value_vec_type &val,
			const traceback_type &tb) {

			if (BatchSize == 1 && traceback_type::max_degree_1) {
				if (direction_type::is_opt(val(0), m_val(0))) {
					m_val = val;
					m_tb.init(tb, xt::ones<bool>({BatchSize}));
				}
			} else {
				m_tb.init(tb, direction_type::opt_q(val, m_val));
				if (!traceback_type::max_degree_1) {
					m_tb.push(tb, xt::equal(val, m_val));
				}

				m_val = direction_type::opt(val, m_val);
			}

			return cont{m_val, m_tb};
		}

		template<typename F>
		inline auto push_many(const F &f) {
			f(cont{m_val, m_tb});
			return cont{m_val, m_tb};
		}

		inline auto add(const value_vec_type &val) {
			m_val += val;
			return cont{m_val, m_tb};
		}

		inline void done() const {
		}
	};

	struct init {
	protected:
		friend class TracingAccumulator;

		inline explicit init(value_vec_type &p_val, traceback_type &p_tb) :
			m_val(p_val), m_tb(p_tb) {
		}

		init(const init&) = delete;
		init& operator=(init const&) = delete;

	public:
		value_vec_type &m_val;
		traceback_type &m_tb;

		inline auto push(
			const value_vec_type &val,
			const index_type u,
			const index_type v) {

			m_val = val;
			m_tb.init(u, v);
			return cont{m_val, m_tb};
		}

		inline auto push(
			const value_vec_type &val,
			const traceback_type &tb) {

			m_val = val;
			m_tb.init(tb, xt::ones<bool>({BatchSize}));
			return cont{m_val, m_tb};
		}
	};

	static inline auto create(
		value_vec_type &p_val,
		traceback_type &p_tb) {

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
		return *m_data;
	}

	inline void pop() {
		m_data.reset();
	}
};

template<typename T, int N, typename Context, int... I>
std::array<T, N> seq_array_impl(const Context &c, std::integer_sequence<int, I...>) {
	return {T(c, I)...};
}

template<typename T, size_t N, typename Context>
auto seq_array(const Context &c) {
	using Sequence = std::make_integer_sequence<int, N>;
	return seq_array_impl<T, N, Context>(c, Sequence{});
}

template<bool Multiple, typename CellType, typename ProblemType, typename Strategy, typename Matrix>
class TracebackIterators;

template<typename CellType, typename ProblemType, typename Strategy, typename Matrix>
class TracebackIterators<false, CellType, ProblemType, Strategy, Matrix> {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;

	constexpr static int BatchSize = CellType::batch_size;

	struct Context {
		const Strategy strategy;
		const Matrix &matrix;
	};

	class Iterator {
	private:
		const Context m_context;
		const int m_batch_index;
		Stack1<std::pair<index_type, index_type>> m_stack;

	public:
		inline Iterator(
			const Context &p_context,
			const int p_batch_index) :

			m_context(p_context),
			m_batch_index(p_batch_index) {
		}

		inline void push(std::pair<index_type, index_type> &&p) {
			m_stack.push(std::move(p));
		}

		template<typename Path>
		inline bool next(Path &p_path) {
			if (m_stack.empty()) {
				return false;
			}

			const auto values = m_context.matrix.template values_n<1, 1>();

			const auto initial_uv = m_stack.top();
			m_stack.pop();

			index_type u = std::get<0>(initial_uv);
			index_type v = std::get<1>(initial_uv);
			const auto best_val = values(u, v)(m_batch_index);

			if (m_context.strategy.has_trace()) { // && m_path.wants_path()
				const auto len_s = m_context.matrix.len_s();
				const auto len_t = m_context.matrix.len_t();

				p_path.begin(len_s, len_t);
				p_path.step(u, v);

				const auto traceback = m_context.matrix.template traceback<1, 1>();

				while (
					m_context.strategy.continue_traceback_1(u, v) &&
					m_context.strategy.continue_traceback_2(values(u, v)(m_batch_index))) {

					const auto &t = traceback(u, v);
					u = t.u(m_batch_index, 0);
					v = t.v(m_batch_index, 0);

                    p_path.step(u, v);
				}
			}

			p_path.emit(best_val);

			return true;
		}
	};

private:
	const Matrix m_matrix;
	std::array<Iterator, BatchSize> m_iterators;

public:
	inline TracebackIterators(
		const Strategy &p_strategy,
		const Matrix &p_matrix,
		const bool /*p_remove_dup*/) :

		m_matrix(p_matrix),
		m_iterators(seq_array<Iterator, BatchSize, Context>(
			Context{p_strategy, m_matrix})) {

	    // p_remove_dup is ignored since we only return 1
	    // alignment in this TracebackIterators implementation.

		p_strategy.seeds(m_matrix).generate(m_iterators);
	}

	inline auto &iterator(const int p_batch_index) {
		return m_iterators[p_batch_index];
	}

	inline const Matrix &matrix() const {
		return m_matrix;
	}
};

template<typename CellType, typename ProblemType, typename Strategy, typename Matrix>
class TracebackIterators<true, CellType, ProblemType, Strategy, Matrix> {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;

	constexpr static int BatchSize = CellType::batch_size;

	struct Context {
		const Strategy strategy;
		const Matrix &matrix;
		const bool remove_dup;
	};

	class Iterator {
	private:
		struct Entry {
			float path_val;
			std::pair<index_type, index_type> current;
			index_type path_len;
		};

		const Context m_context;
		const int m_batch_index;
		std::stack<Entry> m_stack;
		std::unordered_set<
		    CompressedPath<CellType>,
		    CompressedPathHash<CellType>> m_emitted;

        inline bool check_path(const CompressedPath<CellType> &compressed_path) {
            if (m_emitted.find(compressed_path) != m_emitted.end()) {
                return false;
            }
            m_emitted.insert(compressed_path);
            return true;
        }

	public:
		inline Iterator(
			const Context &p_context,
			const int p_batch_index) :

			m_context(p_context),
			m_batch_index(p_batch_index) {

			/*std::cout << "value matrix:" << std::endl;
			m_context.matrix.print_values();
			std::cout << std::endl;

			std::cout << "traceback matrix:" << std::endl;
			m_context.matrix.print_traceback();
			std::cout << std::endl;*/
		}

		inline void push(std::pair<index_type, index_type> &&p0) {
			const std::pair<index_type, index_type> p(p0);

			const index_type u = std::get<0>(p);
			const index_type v = std::get<1>(p);

			const auto values = m_context.matrix.template values_n<1, 1>();
			const auto path_val = values(u, v)(m_batch_index);

			m_stack.push(Entry{
				path_val,
				p,
				0
			});
		}

		template<typename Path>
		inline void init(Path &p_path) {
			if (!m_stack.empty() && m_context.strategy.has_trace()) {
				const auto len_s = m_context.matrix.len_s();
				const auto len_t = m_context.matrix.len_t();
				p_path.begin(len_s, len_t);
			}
		}

		template<typename Path>
		inline bool next(Path &p_path) {
			if (!m_context.strategy.has_trace()) {
				if (m_stack.empty()) {
					return false;
				} else {
					const auto best_val = m_stack.top().path_val;
					m_stack.pop();
					p_path.emit(best_val);
					return true;
				}
			}

			const auto values = m_context.matrix.template values_n<1, 1>();
			const auto traceback = m_context.matrix.template traceback<1, 1>();

			while (!m_stack.empty()) {
				const index_type u1 = std::get<0>(m_stack.top().current);
				const index_type v1 = std::get<1>(m_stack.top().current);

				const auto best_val = m_stack.top().path_val;
				const auto path_len = m_stack.top().path_len;

				m_stack.pop();

				if (path_len == 0) {
					const auto len_s = m_context.matrix.len_s();
					const auto len_t = m_context.matrix.len_t();
					p_path.begin(len_s, len_t);
                } else {
    				p_path.go_back(path_len);
                }

                p_path.step(u1, v1);

				if (m_context.strategy.continue_traceback_1(u1, v1) &&
					m_context.strategy.continue_traceback_2(values(u1, v1)(m_batch_index))) {

					const auto &t = traceback(u1, v1);
					const size_t n = t.size(m_batch_index);
					const index_type path_size = static_cast<index_type>(p_path.size());

					if (n >= 1) {
						for (size_t i = 0; i < n; i++) {
							m_stack.push(Entry{
								best_val,
								{t.u(m_batch_index, i), t.v(m_batch_index, i)},
								path_size
							});
						}
					} else {
						m_stack.push(Entry{
							best_val,
							{no_traceback<index_type>(), no_traceback<index_type>()},
							path_size
						});
					}
				} else {
				    if (!m_context.remove_dup || p_path.check_emit([this] (const auto &path) {
    				    return check_path(path);
				    })) {

                        p_path.emit(best_val);
                        return true;
				    }
				}
			}

			return false;
		}
	};

private:
	const Matrix m_matrix;
	std::array<Iterator, BatchSize> m_iterators;

public:
	inline TracebackIterators(
		const Strategy &p_strategy,
		const Matrix &p_matrix,
		const bool p_remove_dup) :

		m_matrix(p_matrix),
		m_iterators(seq_array<Iterator, BatchSize, Context>(
			Context{p_strategy, m_matrix, p_remove_dup})) {

		p_strategy.seeds(m_matrix).generate(m_iterators);
	}

	inline auto &iterator(const int p_batch_index) {
		return m_iterators[p_batch_index];
	}

	inline const Matrix &matrix() const {
		return m_matrix;
	}
};

template<typename Locality>
struct SharedTracebackIterator {
	typedef typename Locality::cell_type cell_type;
	typedef typename Locality::problem_type problem_type;
	typedef typename Locality::TracebackStrategy traceback_strategy_type;
	typedef Matrix<cell_type, problem_type> layer_matrix_type;

	const MatrixFactoryRef<cell_type, problem_type> factory;

	TracebackIterators<
		!_traceback_type<cell_type, problem_type>::max_degree_1,
		cell_type,
		problem_type,
		traceback_strategy_type,
		layer_matrix_type> iterators;

	inline SharedTracebackIterator(
		const MatrixFactoryRef<cell_type, problem_type> &p_factory,
		const traceback_strategy_type &p_strategy,
		const layer_matrix_type &p_matrix,
		const bool p_remove_dup = false) :

		factory(p_factory),
		iterators(p_strategy, p_matrix, p_remove_dup) {
	}

	auto len_s() const {
		return iterators.matrix().len_s();
	}

	auto len_t() const {
		return iterators.matrix().len_t();
	}
};

template<typename Locality>
using SharedTracebackIteratorRef = std::shared_ptr<SharedTracebackIterator<Locality>>;

template<typename AlignmentFactory, typename Locality>
class AlignmentIterator {
public:
	typedef typename Locality::cell_type cell_type;
	typedef typename Locality::problem_type problem_type;
	typedef typename AlignmentFactory::ref_type alignment_ref_type;

private:
	const SharedTracebackIteratorRef<Locality> m_iterators;
	const int m_batch_index;
	const AlignmentFactory m_alignment_factory;

	typedef typename build_alignment<cell_type, problem_type>::
		template buffered<typename AlignmentFactory::deref_type>
		build_alignment_type;

	build_alignment_type m_build;

public:
	inline AlignmentIterator(
		const SharedTracebackIteratorRef<Locality> &p_iterators,
		const make_path_compressor<cell_type> &p_path_filter_factory,
		const int p_batch_index,
		AlignmentFactory p_alignment_factory = AlignmentFactory()) :

		m_iterators(p_iterators),
		m_batch_index(p_batch_index),
		m_alignment_factory(p_alignment_factory),
		m_build(p_path_filter_factory) {
	}

	alignment_ref_type next() {
		auto &it = m_iterators->iterators.iterator(m_batch_index);
		if (it.next(m_build)) {
			alignment_ref_type alignment = m_alignment_factory.make();
			m_build.copy_to(m_alignment_factory.deref(alignment));
			return alignment;
		} else {
			return alignment_ref_type();
		}
	}
};

template<typename AlignmentFactory, typename SolutionFactory, typename Locality>
class SolutionIterator {
public:
	typedef typename Locality::cell_type cell_type;
	typedef typename Locality::problem_type problem_type;
	typedef typename AlignmentFactory::ref_type alignment_ref_type;
	typedef typename SolutionFactory::ref_type solution_ref_type;

private:
	const SharedTracebackIteratorRef<Locality> m_iterators;
	const int m_batch_index;
	const AlignmentFactory m_alignment_factory;
	const SolutionFactory m_solution_factory;

	typedef build_path<cell_type, problem_type> build_path_type;
	typedef typename build_alignment<cell_type, problem_type>::
		template buffered<typename AlignmentFactory::deref_type>
		build_alignment_type;

	build_multiple<build_path_type, build_alignment_type> m_build;

public:
	inline SolutionIterator(
		const SharedTracebackIteratorRef<Locality> &p_iterators,
		const make_path_compressor<cell_type> &p_path_filter_factory,
		const int p_batch_index,
		const AlignmentFactory p_alignment_factory = AlignmentFactory(),
		const SolutionFactory p_solution_factory = SolutionFactory()) :

		m_iterators(p_iterators),
		m_batch_index(p_batch_index),
		m_alignment_factory(p_alignment_factory),
		m_solution_factory(p_solution_factory),
		m_build(build_path_type(), build_alignment_type(p_path_filter_factory)) {
	}

	solution_ref_type next() {
		auto &it = m_iterators->iterators.iterator(m_batch_index);
		if (it.next(m_build)) {
			solution_ref_type solution_ref = m_solution_factory.make();
			auto &solution = m_solution_factory.deref(solution_ref);

			m_iterators->factory->copy_solution_data(
				m_iterators->len_s(),
				m_iterators->len_t(),
				m_batch_index,
				solution);

			auto alignment = m_alignment_factory.make();
			m_build.template get<1>().copy_to(m_alignment_factory.deref(alignment));
			m_alignment_factory.deref(alignment).set_score(m_build.template get<0>().val());
			solution.set_alignment(alignment);

			solution.set_path(m_build.template get<0>().path());

			//solution.set_algorithm(m_algorithm);

			return solution_ref;
		} else {
			return solution_ref_type();
		}
	}
};

template<typename CellType, typename ProblemType, typename Strategy, typename Matrix>
using TracebackIterators2 = TracebackIterators<
	!_traceback_type<CellType, ProblemType>::max_degree_1, CellType, ProblemType, Strategy, Matrix>;

template<typename Locality, typename Matrix>
inline TracebackIterators2<
	typename Locality::cell_type,
	typename Locality::problem_type,
	typename Locality::TracebackStrategy,
	Matrix>
	make_traceback_iterator(
		const Locality &p_locality,
		Matrix &p_matrix,
		const bool remove_dup = false) {

	return TracebackIterators2<
		typename Locality::cell_type,
		typename Locality::problem_type,
		typename Locality::TracebackStrategy,
		Matrix>(

		typename Locality::TracebackStrategy(p_locality),
		p_matrix,
		remove_dup
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
	typedef Direction direction_type;
	typedef CellType cell_type;

	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef typename CellType::mask_vec_type mask_vec_type;

	constexpr static int BatchSize = CellType::batch_size;

private:
	const value_type worst;
	value_vec_type best_val;
	index_vec_type best_i;
	index_vec_type best_j;

public:
	inline Optima() : worst(direction_type::template worst_val<value_type>()) {
		best_val.fill(worst);
	}

	inline void add(const index_type i, const index_type j, const value_vec_type &val) {
		const mask_vec_type mask = direction_type::opt_q(val, best_val);
		best_val = direction_type::opt(val, best_val);
		best_i = xt::where(mask, i, best_i);
		best_j = xt::where(mask, j, best_j);
	}

	template<typename Stack>
	inline void push(Stack &stack) {
		for (auto k : xt::flatnonzero<xt::layout_type::row_major>(direction_type::opt_q(best_val, worst))) {
			stack[k].push(std::make_pair(best_i(k), best_j(k)));
		}
	}
};

struct LocalInitializers {
};

template<typename CellType, typename ProblemType>
class Local {
public:
	typedef LocalInitializers initializers_type;

	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef typename CellType::mask_vec_type mask_vec_type;

	typedef CellType cell_type;
	typedef ProblemType problem_type;

	constexpr static bool is_global() {
		return false;
	}

	constexpr static value_type ZERO = 0;
	constexpr static int BatchSize = CellType::batch_size;

public:
	typedef TracebackSupport<typename ProblemType::goal_type> tbs_type;
	typedef typename tbs_type::template Accumulator<CellType, ProblemType> accumulator_type;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to(ValueCell &val, TracebackCell &tb) const {
		auto acc = accumulator_type::create(val, tb);
		return acc.push(
			xt::zeros<value_type>({BatchSize}),
			no_traceback<index_type>(),
			no_traceback<index_type>());
	}

	inline Local(const LocalInitializers &p_init) {
	}

	inline const char *name() const {
		return "local";
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<value_type, 1> &p_gap_cost) const {

		for (size_t i = 0; i < p_vector.size(); i++) {
			p_vector(i).fill(ZERO);
		}
	}

	inline value_type zero() const {
		return ZERO;
	}

	template<typename Matrix, typename PathGoal>
	struct TracebackSeeds {
		typedef typename ProblemType::direction_type direction_type;

		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto values = m_matrix.template values_n<1, 1>();

			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			Optima<direction_type, cell_type> optima;

			for (index_type i = len_s - 1; i >= 0; i--) {
				for (index_type j = len_t - 1; j >= 0; j--) {
					optima.add(i, j, values(i, j));
				}
			}

			optima.push(r_seeds);
		}
	};

	template<typename Matrix>
	struct TracebackSeeds<Matrix, goal::path::optimal::all> {
		typedef typename ProblemType::direction_type direction_type;

		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto values = m_matrix.template values_n<1, 1>();

			value_vec_type best_val = xt::zeros<value_type>({BatchSize});

			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			for (index_type i = len_s - 1; i >= 0; i--) {
				for (index_type j = len_t - 1; j >= 0; j--) {
					best_val = direction_type::opt(values(i, j), best_val);
				}
			}

			for (index_type i = len_s - 1; i >= 0; i--) {
				for (index_type j = len_t - 1; j >= 0; j--) {
					const value_vec_type x = values(i, j);
					for (auto k : xt::flatnonzero<xt::layout_type::row_major>(xt::equal(x, best_val))) {
						if (direction_type::is_opt(x(k), ZERO)) {
							r_seeds[k].push(std::make_pair(i, j));
						}
					}
				}
			}
		}
	};

	class TracebackStrategy {
		typedef typename ProblemType::direction_type direction_type;

	public:
		inline TracebackStrategy(
			const Local<CellType, ProblemType> &p_locality) {
		}

		inline constexpr bool has_trace() const {
			return tbs_type::has_trace;
		}

		template<typename Matrix>
		inline auto seeds(const Matrix &p_matrix) const {
			return TracebackSeeds<
				Matrix, typename ProblemType::goal_type::path_goal>{p_matrix};
		}

		inline bool continue_traceback_1(
			const index_type u,
			const index_type v) const {

			return u >= 0 && v >= 0;
		}

		inline bool continue_traceback_2(
			const value_type val) const {

			return direction_type::is_opt(val, ZERO);
		}
	};
};

struct GlobalInitializers {
};

template<typename CellType, typename ProblemType>
class Global {
public:
	typedef GlobalInitializers initializers_type;

	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;

	typedef CellType cell_type;
	typedef ProblemType problem_type;

	static constexpr bool is_global() {
		return true;
	}

	constexpr static int BatchSize = CellType::batch_size;

public:
	typedef TracebackSupport<typename ProblemType::goal_type> tbs_type;
	typedef typename tbs_type::template Accumulator<CellType, ProblemType> accumulator_type;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to(ValueCell &val, TracebackCell &tb) const {
		return accumulator_type::create(val, tb);
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<value_type, 1> &p_gap_cost) const {

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

	class TracebackStrategy {
		typedef typename ProblemType::direction_type direction_type;

	public:
		inline TracebackStrategy(
			const Global<CellType, ProblemType> &p_locality) {
		}

		inline constexpr bool has_trace() const {
			return tbs_type::has_trace;
		}

		template<typename Matrix>
		inline auto seeds(const Matrix &p_matrix) const {
			return TracebackSeeds<
				Matrix, typename ProblemType::goal_type::path_goal>{p_matrix};
		}

		inline bool continue_traceback_1(
			const index_type u,
			const index_type v) const {

			return u >= 0 && v >= 0;
		}

		inline bool continue_traceback_2(
			const value_type val) const {

			return true;
		}
	};
};


struct SemiglobalInitializers {
};

template<typename CellType, typename ProblemType>
class Semiglobal {
public:
	typedef SemiglobalInitializers initializers_type;

	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;

	typedef CellType cell_type;
	typedef ProblemType problem_type;

	static constexpr bool is_global() {
		return false;
	}

	constexpr static int BatchSize = CellType::batch_size;

public:
	typedef TracebackSupport<typename ProblemType::goal_type> tbs_type;
	typedef typename tbs_type::template Accumulator<CellType, ProblemType> accumulator_type;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to(ValueCell &val, TracebackCell &tb) const {
		return accumulator_type::create(val, tb);
	}

	template<typename Vector>
	void init_border_case(
		Vector &&p_vector,
		const xt::xtensor<value_type, 1> &p_gap_cost) const {

		for (size_t i = 0; i < p_vector.size(); i++) {
			p_vector(i).fill(0);
		}
	}

	inline Semiglobal(const SemiglobalInitializers&) {
	}

	template<typename Matrix, typename PathGoal>
	struct TracebackSeeds {
		typedef typename ProblemType::direction_type direction_type;

		const Matrix &m_matrix;

		template<typename Stack>
		void generate(Stack &r_seeds) const {
			const auto len_s = m_matrix.len_s();
			const auto len_t = m_matrix.len_t();

			const auto values = m_matrix.template values_n<1, 1>();

			const index_type last_row = len_s - 1;
			const index_type last_col = len_t - 1;

			Optima<direction_type, cell_type> optima;

			for (index_type j = 0; j < len_t; j++) {
				optima.add(last_row, j, values(last_row, j));
			}

			for (index_type i = 0; i < len_s; i++) {
				optima.add(i, last_col, values(i, last_col));
			}

			optima.push(r_seeds);
		}
	};

	class TracebackStrategy {
		typedef typename ProblemType::direction_type direction_type;

	public:
		inline TracebackStrategy(
			const Semiglobal<CellType, ProblemType> &p_locality) {
		}

		inline constexpr bool has_trace() const {
			return tbs_type::has_trace;
		}

		template<typename Matrix>
		inline auto seeds(const Matrix &p_matrix) const {
			return TracebackSeeds<
				Matrix, typename ProblemType::goal_type::path_goal>{p_matrix};
		}

		inline bool continue_traceback_1(
			const index_type u,
			const index_type v) const {

			return u >= 0 && v >= 0;
		}

		inline bool continue_traceback_2(
			const value_type val) const {

			return true;
		}
	};
};

template<typename CellType, typename ProblemType, typename AlignmentFactory>
class Solution {
public:
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef _traceback_type<CellType, ProblemType> traceback_type;
	typedef typename AlignmentFactory::ref_type alignment_ref_type;

private:
	typedef std::pair<index_type, index_type> coord_type;
	typedef std::pair<coord_type, coord_type> edge_type;

	xt::xtensor<value_type, 3> m_values;
	xt::xtensor<typename traceback_type::Single, 3> m_traceback;
	std::optional<xt::xtensor<index_type, 2>> m_path;
	std::optional<alignment_ref_type> m_alignment;
	AlgorithmMetaDataRef m_algorithm;
	const AlignmentFactory m_alignment_factory;

public:
	inline Solution(const AlignmentFactory p_alignment_factory = AlignmentFactory()) :
		m_alignment_factory(p_alignment_factory) {
	}

	template<typename ValuesMatrix>
	void set_values(
		const ValuesMatrix &p_values,
		const int p_batch_index) {

		const size_t len_k = p_values.shape(0);
		const size_t len_s = p_values.shape(1);
		const size_t len_t = p_values.shape(2);

		xt::xtensor<value_type, 3> values;
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

	inline void set_path(const xt::xtensor<index_type, 2> &p_path) {
		m_path = p_path;
	}

	inline void set_algorithm(const AlgorithmMetaDataRef &p_algorithm) {
		m_algorithm = p_algorithm;
	}

	const xt::xtensor<value_type, 3> &values() const {
		return m_values;
	}

	inline bool has_degree_1_traceback() const {
		return traceback_type::max_degree_1;
	}

	const auto traceback_as_matrix() const {
		const size_t len_k = m_traceback.shape(0);
		const size_t len_s = m_traceback.shape(1);
		const size_t len_t = m_traceback.shape(2);

		xt::xtensor<index_type, 4> traceback;
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

	const std::vector<xt::xtensor<index_type, 3>> traceback_as_edges() const {
		const size_t len_k = m_traceback.shape(0);
		const size_t len_s = m_traceback.shape(1);
		const size_t len_t = m_traceback.shape(2);

		std::vector<xt::xtensor<index_type, 3>> edges;
		edges.resize(len_k);

		const size_t avg_degree = 3; // an estimate
		std::vector<edge_type> layer_edges;
		layer_edges.reserve(len_s * len_t * avg_degree);

		for (size_t k = 0; k < len_k; k++) {
			layer_edges.clear();
			for (size_t i = 0; i < len_s; i++) {
				for (size_t j = 0; j < len_t; j++) {
					const auto &cell = m_traceback(k, i, j);
					const size_t n_edges = cell.size();
					for (size_t q = 0; q < n_edges; q++) {
						layer_edges.emplace_back(std::make_pair<coord_type, coord_type>(
							std::make_pair<index_type, index_type>(
								static_cast<index_type>(i) - 1,
								static_cast<index_type>(j) - 1),
							std::make_pair<index_type, index_type>(
								cell.u(q),
								cell.v(q))));
					}
				}
			}

			xt::xtensor<index_type, 3> &tensor = edges[k];
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

	const std::optional<alignment_ref_type> &alignment() const {
		return m_alignment;
	}

	void set_alignment(const alignment_ref_type &p_alignment) {
		m_alignment = p_alignment;
	}

	const std::optional<value_type> score() const {
		if (m_alignment.has_value()) {
			return m_alignment_factory.deref(*m_alignment).score();
		} else {
			return std::optional<value_type>();
		}
	}

	const auto &algorithm() const {
		return m_algorithm;
	}
};

template<typename CellType, typename ProblemType, typename AlignmentFactory>
using SolutionRef = std::shared_ptr<Solution<CellType, ProblemType, AlignmentFactory>>;


struct matrix_name {
	constexpr static int D = 0;
	constexpr static int P = 1;
	constexpr static int Q = 2;
};

template<typename T>
struct SharedPtrFactory {
	typedef T deref_type;
	typedef std::shared_ptr<T> ref_type;

	inline T &deref(const std::shared_ptr<T> &p) const {
		return *p.get();
	}

	inline std::shared_ptr<T> make() const {
		return std::make_shared<T>();
	}
};

template<typename CellType, typename ProblemType, template<typename, typename> class Locality>
class Solver {
public:
	typedef CellType cell_type;
	typedef ProblemType problem_type;

	typedef typename CellType::value_type value_type;
	typedef typename CellType::value_vec_type value_vec_type;
	typedef typename CellType::index_type index_type;
	typedef typename CellType::index_vec_type index_vec_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef Locality<CellType, ProblemType> locality_type;

protected:
	const Locality<CellType, ProblemType> m_locality;
	const MatrixFactoryRef<CellType, ProblemType> m_factory;
	const AlgorithmMetaDataRef m_algorithm;

	template<typename ValueCell, typename TracebackCell>
	inline auto accumulate_to_nolocal(ValueCell &val, TracebackCell &tb) const {
		typedef TracebackSupport<typename ProblemType::goal_type> tbs_type;
		typedef typename tbs_type::template Accumulator<CellType, ProblemType> accumulator_type;

		return accumulator_type::create(val, tb);
	}

public:
	inline Solver(
		const typename Locality<CellType, ProblemType>::initializers_type &p_locality_init,
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const size_t p_layer_count,
		const AlgorithmMetaDataRef &p_algorithm) :

		m_locality(p_locality_init),
		m_factory(std::make_shared<MatrixFactory<CellType, ProblemType>>(
			p_max_len_s, p_max_len_t, p_layer_count)),
		m_algorithm(p_algorithm) {
	}

	virtual inline ~Solver() {
	}

	inline index_type max_len_s() const {
		return m_factory->max_len_s();
	}

	inline index_type max_len_t() const {
		return m_factory->max_len_t();
	}

	inline constexpr int batch_size() const {
		return CellType::batch_size;
	}

	template<int Layer>
	inline auto matrix(const index_type len_s, const index_type len_t) {
		return m_factory->make<Layer>(len_s, len_t);
	}

	inline value_vec_type score(
		const index_vec_type &len_s,
		const index_vec_type &len_t) const {

		value_vec_type scores;
		build_val<CellType, ProblemType> val_only;

		for (int i = 0; i < CellType::batch_size; i++) {
			auto matrix = m_factory->template make<matrix_name::D>(len_s(i), len_t(i));
			auto tb = make_traceback_iterator(m_locality, matrix, true);

			const bool tb_good = tb.iterator(i).next(val_only);
			scores(i) = tb_good ? val_only.val() : direction_type::template worst_val<value_type>();
		}
		return scores;
	}

	template<typename AlignmentFactory>
	inline void alignment(
		const index_vec_type &len_s,
		const index_vec_type &len_t,
		std::array<typename AlignmentFactory::ref_type, CellType::batch_size> &alignments,
		const AlignmentFactory alignment_factory = AlignmentFactory()) const {

		for (int i = 0; i < CellType::batch_size; i++) {
			auto matrix = m_factory->template make<matrix_name::D>(len_s(i), len_t(i));
			auto tb = make_traceback_iterator(m_locality, matrix, true);
			alignments[i] = alignment_factory.make();
			auto &alignment = alignment_factory.deref(alignments[i]);
			auto build = typename build_alignment<CellType, ProblemType>::
				template unbuffered<typename AlignmentFactory::deref_type>(
				    alignment,
				    make_path_compressor<CellType>());
			if (!tb.iterator(i).next(build)) {
				alignment.set_score(direction_type::template worst_val<value_type>());
			}
		}
	}

	template<typename AlignmentFactory>
	std::vector<std::shared_ptr<AlignmentIterator<
			AlignmentFactory, Locality<CellType, ProblemType>>>> alignment_iterator(
		const index_vec_type &len_s,
		const index_vec_type &len_t,
		const bool remove_duplicate_alignments,
		const AlignmentFactory alignment_factory = AlignmentFactory()) const {

		std::vector<std::shared_ptr<AlignmentIterator<
			AlignmentFactory, Locality<CellType, ProblemType>>>> iterators;
		iterators.reserve(CellType::batch_size);

		const auto detached_factory = m_factory->template copy<matrix_name::D>(
		    xt::amax(len_s)(), xt::amax(len_t)());

		for (int i = 0; i < CellType::batch_size; i++) {
			auto matrix = detached_factory->template make<0>(len_s(i), len_t(i));
			auto shared_it = std::make_shared<SharedTracebackIterator<
				Locality<CellType, ProblemType>>>(
				    detached_factory, m_locality, matrix, remove_duplicate_alignments);

			iterators.push_back(std::make_shared<AlignmentIterator<
				AlignmentFactory, Locality<CellType, ProblemType>>>(
					shared_it,
					make_path_compressor<CellType>(),
					i,
					alignment_factory
				));
		}

		return iterators;
	}

	template<typename AlignmentFactory, typename SolutionFactory>
	void solution(
		const index_vec_type &len_s,
		const index_vec_type &len_t,
		std::array<typename SolutionFactory::ref_type, CellType::batch_size> &solutions,
		AlignmentFactory alignment_factory = AlignmentFactory(),
		SolutionFactory solution_factory = SolutionFactory()) const {

		for (int i = 0; i < CellType::batch_size; i++) {
			auto matrix = m_factory->template make<matrix_name::D>(len_s(i), len_t(i));
			auto tb = make_traceback_iterator(m_locality, matrix, true);

			solutions[i] = solution_factory.make();
			auto &solution = solution_factory.deref(solutions[i]);

			m_factory->copy_solution_data(len_s(i), len_t(i), i, solution);

			auto alignment = alignment_factory.make();

			typedef build_path<CellType, ProblemType> build_path_type;
			typedef typename build_alignment<CellType, ProblemType>::
				template unbuffered<typename AlignmentFactory::deref_type>
				build_alignment_type;

			auto build = build_multiple<build_path_type, build_alignment_type>(
				build_path_type(),
				build_alignment_type(
				    alignment_factory.deref(alignment),
				    make_path_compressor<CellType>())
			);

			const bool tb_good = tb.iterator(i).next(build);
			if (tb_good) {
				alignment_factory.deref(alignment).set_score(build.template get<0>().val());
				solution.set_path(build.template get<0>().path());
				solution.set_alignment(alignment);
			}

			solution.set_algorithm(m_algorithm);
		}
	}

	template<typename AlignmentFactory, typename SolutionFactory>
	std::vector<std::shared_ptr<SolutionIterator<
			AlignmentFactory, SolutionFactory, Locality<CellType, ProblemType>>>> solution_iterator(
		const index_vec_type &len_s,
		const index_vec_type &len_t,
		const bool remove_duplicate_alignments,
		const AlignmentFactory alignment_factory = AlignmentFactory(),
		const SolutionFactory solution_factory = SolutionFactory()) const {

		std::vector<std::shared_ptr<SolutionIterator<
			AlignmentFactory, SolutionFactory, Locality<CellType, ProblemType>>>> iterators;
		iterators.reserve(CellType::batch_size);

		const auto detached_factory = m_factory->template copy<matrix_name::D>(
		    xt::amax(len_s)(), xt::amax(len_t)());

		for (int i = 0; i < CellType::batch_size; i++) {
			auto matrix = detached_factory->template make<0>(len_s(i), len_t(i));
			auto shared_it = std::make_shared<SharedTracebackIterator<
				Locality<CellType, ProblemType>>>(
				    detached_factory, m_locality, matrix, remove_duplicate_alignments);

			iterators.push_back(std::make_shared<SolutionIterator<
				AlignmentFactory, SolutionFactory, Locality<CellType, ProblemType>>>(
					shared_it,
					make_path_compressor<CellType>(),
					i,
					alignment_factory,
					solution_factory
				));
		}

		return iterators;
	}
};

template<typename CellType, typename ProblemType, template<typename, typename> class Locality>
using AlignmentSolver = Solver<CellType, ProblemType, Locality>;

template<typename CellType, typename ProblemType, template<typename, typename> class Locality>
class LinearGapCostSolver final : public AlignmentSolver<CellType, ProblemType, Locality> {
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
	typedef typename ProblemType::goal_type goal_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename Locality<CellType, ProblemType>::initializers_type locality_init_type;

private:
	const value_type m_gap_cost_s;
	const value_type m_gap_cost_t;

public:
	typedef value_type gap_cost_spec_type;

	inline LinearGapCostSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const value_type p_gap_cost_s,
		const value_type p_gap_cost_t,
		const locality_init_type p_locality_init = locality_init_type()) :

		AlignmentSolver<CellType, ProblemType, Locality>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			1, // layer count
			std::make_shared<AlgorithmMetaData>(
				Locality<CellType, ProblemType>::is_global() ?
					"Needleman-Wunsch": "Smith-Waterman",
				"n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto values = this->m_factory->template values<matrix_name::D>();
		constexpr value_type gap_sgn = direction_type::is_minimize() ? 1 : -1;

		this->m_locality.init_border_case(
			xt::view(values, xt::all(), 0),
			xt::arange<index_type>(0, p_max_len_s + 1) * p_gap_cost_s * gap_sgn);
		this->m_locality.init_border_case(
			xt::view(values, 0, xt::all()),
			xt::arange<index_type>(0, p_max_len_t + 1) * p_gap_cost_t * gap_sgn);
	}

	inline value_type gap_cost_s(const size_t len) const {
		return m_gap_cost_s * len;
	}

	inline value_type gap_cost_t(const size_t len) const {
		return m_gap_cost_t * len;
	}

	template<typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = this->m_factory->template make<matrix_name::D>(len_s, len_t);

		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();
		constexpr value_type gap_sgn = direction_type::is_minimize() ? 1 : -1;

		for (index_type u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (index_type v = 0; static_cast<size_t>(v) < len_t; v++) {

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

	inline Value at(const size_t k) const {
		return u * k + v;
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
class AffineGapCostSolver final : public AlignmentSolver<CellType, ProblemType, Locality> {
public:
	// Gotoh, O. (1982). An improved algorithm for matching biological sequences.
	// Journal of Molecular Biology, 162(3), 705–708. https://doi.org/10.1016/0022-2836(82)90398-9

	typedef typename ProblemType::goal_type goal_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename Locality<CellType, ProblemType>::initializers_type locality_init_type;

	typedef AffineCost<value_type> cost_type;

private:
	const cost_type m_gap_cost_s;
	const cost_type m_gap_cost_t;

public:
	inline AffineGapCostSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const cost_type &p_gap_cost_s,
		const cost_type &p_gap_cost_t,
		const locality_init_type p_locality_init = locality_init_type()) :

		AlignmentSolver<CellType, ProblemType, Locality>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			3, // layer count
			std::make_shared<AlgorithmMetaData>("Gotoh", "n^2", "n^2")),
		m_gap_cost_s(p_gap_cost_s),
		m_gap_cost_t(p_gap_cost_t) {

		auto matrix_D = this->m_factory->template make<matrix_name::D>(p_max_len_s, p_max_len_t);
		auto matrix_P = this->m_factory->template make<matrix_name::P>(p_max_len_s, p_max_len_t);
		auto matrix_Q = this->m_factory->template make<matrix_name::Q>(p_max_len_s, p_max_len_t);

		auto D = matrix_D.template values<0, 0>();
		auto P = matrix_P.template values<0, 0>();
		auto Q = matrix_Q.template values<0, 0>();

		const auto inf = std::numeric_limits<value_type>::infinity() * (direction_type::is_minimize() ? 1 : -1);

		for (auto &cell : xt::view(Q, xt::all(), 0)) {
			cell.fill(inf);
		}
		for (auto &cell : xt::view(P, 0, xt::all())) {
			cell.fill(inf);
		}

		constexpr value_type gap_sgn = direction_type::is_minimize() ? 1 : -1;

		// setting D(m, 0) = P(m, 0) = w(m)
		this->m_locality.init_border_case(
			xt::view(D, xt::all(), 0),
			m_gap_cost_s.vector(p_max_len_s + 1) * gap_sgn);
		this->m_locality.init_border_case(
			xt::view(P, xt::all(), 0),
			m_gap_cost_s.vector(p_max_len_s + 1) * gap_sgn);

		// setting D(0, n) = Q(0, n) = w(n)
		this->m_locality.init_border_case(
			xt::view(D, 0, xt::all()),
			m_gap_cost_t.vector(p_max_len_t + 1) * gap_sgn);
		this->m_locality.init_border_case(
			xt::view(Q, 0, xt::all()),
			m_gap_cost_t.vector(p_max_len_t + 1) * gap_sgn);

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

		auto matrix_D = this->m_factory->template make<matrix_name::D>(len_s, len_t);
		auto matrix_P = this->m_factory->template make<matrix_name::P>(len_s, len_t);
		auto matrix_Q = this->m_factory->template make<matrix_name::Q>(len_s, len_t);

		auto D = matrix_D.template values_n<1, 1>();
		auto tb_D = matrix_D.template traceback_n<1, 1>();
		auto P = matrix_P.template values_n<1, 1>();
		auto tb_P = matrix_P.template traceback_n<1, 1>();
		auto Q = matrix_Q.template values_n<1, 1>();
		auto tb_Q = matrix_Q.template traceback_n<1, 1>();

		constexpr value_type gap_sgn = direction_type::is_minimize() ? 1 : -1;

		for (index_type i = 0; static_cast<size_t>(i) < len_s; i++) {

			for (index_type j = 0; static_cast<size_t>(j) < len_t; j++) {

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

	inline value_type gap_cost_s(const size_t len) const {
		return m_gap_cost_s.at(len);
	}

	inline value_type gap_cost_t(const size_t len) const {
		return m_gap_cost_t.at(len);
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
class GeneralGapCostSolver final : public AlignmentSolver<CellType, ProblemType, Locality> {
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
	typedef typename ProblemType::goal_type goal_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;
	typedef typename Locality<CellType, ProblemType>::initializers_type locality_init_type;

private:
	struct GapCost {
		inline GapCost(
			const xt::xtensor<value_type, 1> &s_,
			const xt::xtensor<value_type, 1> &t_) : s(s_), t(t_) {
		}

		xt::xtensor<value_type, 1> s;
		xt::xtensor<value_type, 1> t;
	};

	const std::unique_ptr<GapCost> m_gap_cost;

public:
	typedef GapTensorFactory<value_type> GapCostSpec;

	inline GeneralGapCostSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t,
		const GapTensorFactory<value_type> &p_gap_cost_s,
		const GapTensorFactory<value_type> &p_gap_cost_t,
		const locality_init_type p_locality_init = locality_init_type()) :

		AlignmentSolver<CellType, ProblemType, Locality>(
			p_locality_init,
			p_max_len_s,
			p_max_len_t,
			1, // layer count
			std::make_shared<AlgorithmMetaData>("Waterman-Smith-Beyer", "n^3", "n^2")),
		m_gap_cost(std::make_unique<GapCost>(
			p_gap_cost_s(p_max_len_s + 1),
			p_gap_cost_t(p_max_len_t + 1)
		)) {

		check_gap_tensor_shape(m_gap_cost->s, p_max_len_s + 1);
		check_gap_tensor_shape(m_gap_cost->t, p_max_len_t + 1);

		auto values = this->m_factory->template values<matrix_name::D>();
		constexpr value_type gap_sgn = direction_type::is_minimize() ? 1 : -1;

		this->m_locality.init_border_case(
			xt::view(values, xt::all(), 0),
			m_gap_cost->s * gap_sgn);

		this->m_locality.init_border_case(
			xt::view(values, 0, xt::all()),
			m_gap_cost->t * gap_sgn);
	}

	inline value_type gap_cost_s(const size_t len) const {
		PPK_ASSERT(len < m_gap_cost->s.shape(0));
		return m_gap_cost->s(len);
	}

	inline value_type gap_cost_t(const size_t len) const {
		PPK_ASSERT(len < m_gap_cost->t.shape(0));
		return m_gap_cost->t(len);
	}

	template<typename Pairwise>
	void solve(
		const Pairwise &pairwise,
		const size_t len_s,
		const size_t len_t) const {

		auto matrix = this->m_factory->template make<matrix_name::D>(len_s, len_t);

		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();
		constexpr value_type gap_sgn = direction_type::is_minimize() ? 1 : -1;

		const auto &gap_cost_s = this->m_gap_cost->s;
		const auto &gap_cost_t = this->m_gap_cost->t;

		for (index_type u = 0; static_cast<size_t>(u) < len_s; u++) {

			for (index_type v = 0; static_cast<size_t>(v) < len_t; v++) {

				this->m_locality.accumulate_to(
					values(u, v), traceback(u, v))

				.push(
					values(u - 1, v - 1) + pairwise(u, v),
					u - 1, v - 1)

				.push_many([u, v, gap_cost_s, gap_sgn, &values] (auto acc) {
					for (index_type k = -1; k < u; k++) {
						acc.push(
							values(k, v) + gap_cost_s(u - k) * gap_sgn,
							k, v);
					}
				})

				.push_many([u, v, gap_cost_t, gap_sgn, &values] (auto acc) {
					for (index_type k = -1; k < v; k++) {
						acc.push(
							values(u, k) + gap_cost_t(v - k) * gap_sgn,
							u, k);
					}
				})

				.done();
			}
		}
	}
};

template<typename CellType, typename ProblemType>
class DynamicTimeSolver final : public Solver<CellType, ProblemType, Global> {
public:
	typedef typename ProblemType::goal_type goal_type;
	typedef typename ProblemType::direction_type direction_type;
	typedef typename CellType::index_type index_type;
	typedef typename CellType::value_type value_type;

	inline DynamicTimeSolver(
		const size_t p_max_len_s,
		const size_t p_max_len_t) :

		Solver<CellType, ProblemType, Global>(
			GlobalInitializers(),
			p_max_len_s,
			p_max_len_t,
			1, // layer count
			std::make_shared<AlgorithmMetaData>("DTW", "n^2", "n^2")) {

		auto values = this->m_factory->template values<matrix_name::D>();
		const auto worst = std::numeric_limits<value_type>::infinity() * (direction_type::is_minimize() ? 1 : -1);
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

		auto matrix = this->m_factory->template make<matrix_name::D>(len_s, len_t);
		auto values = matrix.template values_n<1, 1>();
		auto traceback = matrix.template traceback<1, 1>();

		for (index_type u = 0; static_cast<size_t>(u) < len_s; u++) {
			for (index_type v = 0; static_cast<size_t>(v) < len_t; v++) {

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

	inline value_type gap_cost_s(const size_t len) const {
		return 0;
	}

	inline value_type gap_cost_t(const size_t len) const {
		return 0;
	}
};

} // namespace core
} // namespace pyalign

#endif // __PYALIGN_SOLVER_H__
