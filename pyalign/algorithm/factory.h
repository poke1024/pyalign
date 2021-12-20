#ifndef __PYALIGN_FACTORY_H__
#define __PYALIGN_FACTORY_H__ 1

#include "pyalign/algorithm/solver.h"
#include "pyalign/algorithm/enum.h"

namespace pyalign {

namespace py = pybind11;

template<typename Value>
xt::xtensor<Value, 1> zero_gap_tensor(const size_t p_len) {
	xt::xtensor<Value, 1> w;
	w.resize({p_len});
	w.fill(0);
	return w;
}

template<typename Value>
core::GapTensorFactory<Value> to_gap_tensor_factory(const py::object &p_gap) {
	if (p_gap.is_none()) {
		return zero_gap_tensor<Value>;
	} else {
		const auto f = p_gap.attr("costs").cast<std::function<xt::pytensor<Value, 1>(size_t)>>();
		return [f] (const size_t n) {
			py::gil_scoped_acquire acquire;
			const xt::xtensor<Value, 1> costs = f(n);
			return costs;
		};
	}
}

template<typename Value>
struct GapCostOptions {
	inline GapCostOptions(const py::object &p_gap) {
		if (p_gap.is_none()) {
			linear = 0.0f;
		} else {
	        const py::dict cost = p_gap.attr("to_special_case")().cast<py::dict>();

	        if (cost.contains("affine")) {
	            // we flip u and v here:
				//
	            // * we get (u, v) that specifies w(k) = u + v k, which is the formulation in
	            // pyalign.gaps (in accordance with Stojmirović et al. and others)
	            // * AffineCost takes (u, v) as w(k) = u k + v, which is Gotoh's formulation

	            auto affine_tuple = cost["affine"].cast<py::tuple>();
				affine = core::AffineCost<Value>(
					affine_tuple[1].cast<Value>(),
					affine_tuple[0].cast<Value>()
				);

	        } else if (cost.contains("linear")) {
	            linear = cost["linear"].cast<Value>();
	        } else {
	            general = to_gap_tensor_factory<Value>(p_gap);
	        }
	    }
	}

	std::optional<Value> linear;
	std::optional<core::AffineCost<Value>> affine;
	std::optional<core::GapTensorFactory<Value>> general;
};

template<typename Value>
auto to_gap_cost_options(const py::object p_gap_cost) {
	py::object gap_s = py::none();
	py::object gap_t = py::none();

	if (py::isinstance<py::dict>(p_gap_cost)) {
		const py::dict gap_cost_dict = p_gap_cost.cast<py::dict>();

		if (gap_cost_dict.contains("s")) {
			gap_s = gap_cost_dict["s"];
		}
		if (gap_cost_dict.contains("t")) {
			gap_t = gap_cost_dict["t"];
		}
	} else {
		gap_s = p_gap_cost;
		gap_t = p_gap_cost;
	}

	return std::make_pair<GapCostOptions<Value>>(
		GapCostOptions<Value>(gap_s),
		GapCostOptions<Value>(gap_t)
	);
}

template<typename Value>
class GapCosts {
	const std::pair<GapCostOptions<Value>, GapCostOptions<Value>> m_options;

public:
	inline GapCosts(const py::object p_gap_cost) : m_options(
		to_gap_cost_options<Value>(p_gap_cost)) {
	}

	const GapCostOptions<Value> &s() const {
		return std::get<0>(m_options);
	}

	const GapCostOptions<Value> &t() const {
		return std::get<1>(m_options);
	}
};

template<typename Index>
xt::pytensor<Index, 1> invert(
	const xt::pytensor<Index, 1> &p_source,
	const size_t p_inverted_len) {

	xt::pytensor<Index, 1> inverted;
	inverted.resize({static_cast<ssize_t>(p_inverted_len)});
	inverted.fill(-1);

	const size_t n = p_source.shape(0);
	for (size_t i = 0; i < n; i++) {
		auto j = p_source(i);
		if (j >= 0) {
			inverted(j) = i;
		}
	}

	return inverted;
}

template<typename Index>
class Alignment {
private:
	std::optional<xt::pytensor<Index, 1>> m_s_to_t;
	std::optional<xt::pytensor<Index, 1>> m_t_to_s;
	Index m_len_s;
	Index m_len_t;
	float m_score;

public:
	inline Alignment() : m_len_s(0), m_len_t(0) {
	}

	inline void resize(const size_t len_s, const size_t len_t) {
		m_len_s = len_s;
		m_len_t = len_t;

		if (len_s <= len_t) {
			m_s_to_t = xt::pytensor<Index, 1>();
			(*m_s_to_t).resize({static_cast<ssize_t>(len_s)});
			(*m_s_to_t).fill(-1);
		} else {
			m_t_to_s = xt::pytensor<Index, 1>();
			(*m_t_to_s).resize({static_cast<ssize_t>(len_t)});
			(*m_t_to_s).fill(-1);
		}
	}

	inline void add_edge(const size_t u, const size_t v) {
		if (m_s_to_t.has_value()) {
			(*m_s_to_t)[u] = v;
		} else {
			(*m_t_to_s)[v] = u;
		}
	}

	inline void set_score(const float p_score) {
		m_score = p_score;
	}

	inline float score() const {
		return m_score;
	}

	inline const xt::pytensor<Index, 1> &s_to_t() {
		if (!m_s_to_t.has_value()) {
			m_s_to_t = invert<Index>(*m_t_to_s, m_len_s);
		}
		return *m_s_to_t;
	}

	inline const xt::pytensor<Index, 1> &t_to_s() {
		if (!m_t_to_s.has_value()) {
			m_t_to_s = invert<Index>(*m_s_to_t, m_len_t);
		}
		return *m_t_to_s;
	}
};

template<typename Index>
using AlignmentRef = std::shared_ptr<Alignment<Index>>;

template<typename CellType, typename ProblemType>
using NativeSolution = core::Solution<
	CellType,
	ProblemType,
	core::SharedPtrFactory<Alignment<typename CellType::index_type>>>;

template<typename Index>
struct SharedAlignment {
	typedef std::shared_ptr<Alignment<Index>> ref_type;
	typedef Alignment<Index> deref_type;

	static inline Alignment<Index> &deref(const std::shared_ptr<Alignment<Index>> &p_ref) {
		return *p_ref.get();
	}

	static inline std::shared_ptr<Alignment<Index>> make() {
		return std::make_shared<Alignment<Index>>();
	}
};

template<typename Index>
class AlignmentIterator {
public:
	virtual ~AlignmentIterator() {
	}

	virtual AlignmentRef<Index> next() = 0;
};

template<typename Index>
using AlignmentIteratorRef = std::shared_ptr<AlignmentIterator<Index>>;

template<typename Index, typename Locality>
class AlignmentIteratorImpl : public AlignmentIterator<Index> {
	const std::shared_ptr<core::AlignmentIterator<
		core::SharedPtrFactory<Alignment<Index>>, Locality>> m_iterator;

public:
	inline AlignmentIteratorImpl(
		const std::shared_ptr<core::AlignmentIterator<
			core::SharedPtrFactory<Alignment<Index>>, Locality>> &p_iterator) :
		m_iterator(p_iterator) {
	}

	virtual AlignmentRef<Index> next() override {
		return m_iterator->next();
	}
};

typedef core::AlgorithmMetaData Algorithm;
typedef core::AlgorithmMetaDataRef AlgorithmRef;

class Solution {
public:
	typedef float Value;
	typedef int16_t Index;

	virtual ~Solution() {
	}

	virtual xt::pytensor<Value, 3> values() const = 0;
	virtual bool traceback_has_max_degree_1() const = 0;
	virtual xt::pytensor<Index, 4> traceback_as_matrix() const = 0;
	virtual py::list traceback_as_edges() const = 0;
	virtual AlgorithmRef algorithm() const = 0;
	virtual py::object score() const = 0;
	virtual py::object alignment() const = 0;
	virtual py::object path() const = 0;
};

typedef std::shared_ptr<Solution> SolutionRef;

class SolutionIterator {
public:
	virtual ~SolutionIterator() {
	}

	virtual SolutionRef next() = 0;
};

typedef std::shared_ptr<SolutionIterator> SolutionIteratorRef;


template<typename CellType, typename ProblemType>
class SolutionImpl : public Solution {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;

private:
	const core::SolutionRef<CellType, ProblemType, core::SharedPtrFactory<Alignment<Index>>> m_solution;

public:
	inline SolutionImpl(
		const core::SolutionRef<CellType, ProblemType, core::SharedPtrFactory<Alignment<Index>>> &p_solution) :

		m_solution(p_solution) {
	}

	virtual bool traceback_has_max_degree_1() const override {
		return m_solution->has_degree_1_traceback();
	}

	virtual xt::pytensor<Value, 3> values() const override {
		return m_solution->values();
	}

	virtual xt::pytensor<Index, 4> traceback_as_matrix() const override {
		return m_solution->traceback_as_matrix();
	}

	virtual py::list traceback_as_edges() const override {
		const auto edges = m_solution->traceback_as_edges();

		py::list py_edges;
		for (const auto &layer_edges : edges) {
			py_edges.append(xt::pytensor<Index, 3>(layer_edges));
		}

		return py_edges;
	}

	virtual AlgorithmRef algorithm() const override {
		return m_solution->algorithm();
	}

	virtual py::object score() const override {
		const auto score = m_solution->score();
		if (score.has_value()) {
			return py::cast(*score);
		} else {
			return py::none();
		}
	}

	virtual py::object alignment() const override {
		if (m_solution->alignment().has_value()) {
			return py::cast(*m_solution->alignment());
		} else {
			return py::none();
		}
	}

	virtual py::object path() const override {
		if (m_solution->path().has_value()) {
			const xt::pytensor<Index, 2> p = *m_solution->path();
			return p;
		} else {
			return py::none();
		}
	}
};

template<typename Locality>
class SolutionIteratorImpl : public SolutionIterator {
public:
	typedef typename Locality::cell_type CellType;
	typedef typename Locality::problem_type ProblemType;

	typedef typename CellType::index_type Index;

private:
	const std::shared_ptr<core::SolutionIterator<
		core::SharedPtrFactory<Alignment<Index>>,
		core::SharedPtrFactory<NativeSolution<CellType, ProblemType>>,
		Locality>> m_iterator;

public:
	inline SolutionIteratorImpl(
		const std::shared_ptr<core::SolutionIterator<
			core::SharedPtrFactory<Alignment<Index>>,
			core::SharedPtrFactory<NativeSolution<CellType, ProblemType>>,
			Locality>> &p_iterator) :
		m_iterator(p_iterator) {
	}

	virtual SolutionRef next() override {
		const auto r = m_iterator->next();
		if (r.get()) {
			return std::make_shared<SolutionImpl<CellType, ProblemType>>(r);
		} else {
			return SolutionRef();
		}
	}
};

inline void check_batch_size(const size_t given, const size_t batch_size) {
	if (given != batch_size) {
		std::ostringstream err;
		err << "problem of batch size " << given <<
			" given to solver that was configured to batch size " << batch_size;
		throw std::invalid_argument(err.str());
	}
}

template<typename Array, std::size_t... T>
py::tuple to_tuple_with_seq(const Array& array, std::index_sequence<T...>) {
    return py::make_tuple(array[T]...);
}

template<typename T, int N>
py::tuple to_tuple(const std::array<T, N> &obj) {
	py::object py_obj[N];

	for (int i = 0; i < N; i++) {
		py_obj[i] = py::cast(obj[i]);
	}

	return to_tuple_with_seq(
		py_obj,
		std::make_index_sequence<N>());
}

template<typename Value, typename Index>
class Solver {
public:
	virtual inline ~Solver() {
	}

	virtual py::dict options() const = 0;

	virtual int batch_size() const = 0;

	virtual xt::pytensor<Value, 1> solve_for_score(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual xt::pytensor<Value, 1> solve_indexed_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual xt::pytensor<Value, 1> solve_binary_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_for_alignment(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_for_alignment_iterator(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_alignment_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_binary_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_binary_for_alignment_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_for_solution(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_for_solution_iterator(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_solution_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_binary_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const = 0;

	virtual py::tuple solve_binary_for_solution_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const = 0;
};

template<typename Value, typename Index>
using SolverRef = std::shared_ptr<Solver<Value, Index>>;

template<typename CellType>
struct base_matrix_form {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<Index, 2> &m_length;

	inline base_matrix_form(
	    const xt::pytensor<Index, 2> &p_length) : m_length(p_length) {
	}

	inline void check() const {
		if (m_length.shape(0) != 2 || m_length.shape(1) != CellType::batch_size) {
			std::ostringstream err;
			err << "m_length has shape (" << m_length.shape(0) << ", " <<
				m_length.shape(1) << "), expected (2, " << CellType::batch_size << ")";
			throw std::invalid_argument(err.str());
		}
	}

	inline IndexVec len_s() const {
		return xt::view(m_length, 0, xt::all());
	}

	inline IndexVec len_t() const {
		return xt::view(m_length, 1, xt::all());
	}
};

template<typename CellType>
struct matrix_form : base_matrix_form<CellType> {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<Value, 3> &m_similarity;

	inline matrix_form(
	    const xt::pytensor<Value, 3> &p_similarity,
	    const xt::pytensor<Index, 2> &p_length) :
	    base_matrix_form<CellType>(p_length),
	    m_similarity(p_similarity) {
	}

	inline void check() const {
		check_batch_size(m_similarity.shape(2), CellType::batch_size);
		base_matrix_form<CellType>::check();
	}

	inline size_t batch_len_s() const {
		return m_similarity.shape(0);
	}

	inline size_t batch_len_t() const {
		return m_similarity.shape(1);
	}

	inline ValueVec operator()(const Index i, const Index j) const {
		ValueVec v = xt::view(m_similarity, i, j, xt::all());
		return v;
	}
};

template<typename CellType>
struct indexed_matrix_form : base_matrix_form<CellType> {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<uint32_t, 2> &m_a;
	const xt::pytensor<uint32_t, 2> &m_b;
	const xt::pytensor<Value, 2> &m_similarity;

	inline indexed_matrix_form(
	    const xt::pytensor<uint32_t, 2> &p_a,
    	const xt::pytensor<uint32_t, 2> &p_b,
	    const xt::pytensor<Value, 2> &p_similarity,
	    const xt::pytensor<Index, 2> &p_length) :

	    base_matrix_form<CellType>(p_length),
	    m_a(p_a),
	    m_b(p_b),
	    m_similarity(p_similarity) {
	}

	inline void check() const {
		check_batch_size(m_a.shape(0), CellType::batch_size);
		check_batch_size(m_b.shape(0), CellType::batch_size);

		if (xt::amax(m_a)() >= m_similarity.shape(0)) {
			throw std::invalid_argument("out of bounds index in a");
		}
		if (xt::amax(m_b)() >= m_similarity.shape(1)) {
			throw std::invalid_argument("out of bounds index in b");
		}

		base_matrix_form<CellType>::check();
	}

	inline size_t batch_len_s() const {
		return m_a.shape(1);
	}

	inline size_t batch_len_t() const {
		return m_b.shape(1);
	}

	inline ValueVec operator()(const Index i, const Index j) const {
		ValueVec v;
		for (int k = 0; k < CellType::batch_size; k++) {
			v(k) = m_similarity(m_a(k, i), m_b(k, j));
		}
		return v;
	}
};

template<typename CellType>
struct binary_matrix_form : base_matrix_form<CellType> {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<uint32_t, 2> &m_a;
	const xt::pytensor<uint32_t, 2> &m_b;
    const Value m_eq;
    const Value m_ne;

	inline binary_matrix_form(
	    const xt::pytensor<uint32_t, 2> &p_a,
    	const xt::pytensor<uint32_t, 2> &p_b,
    	const Value p_eq,
    	const Value p_ne,
	    const xt::pytensor<Index, 2> &p_length) :

	    base_matrix_form<CellType>(p_length),
	    m_a(p_a),
	    m_b(p_b),
	    m_eq(p_eq),
	    m_ne(p_ne) {
	}

	inline void check() const {
		check_batch_size(m_a.shape(0), CellType::batch_size);
		check_batch_size(m_b.shape(0), CellType::batch_size);
		base_matrix_form<CellType>::check();
	}

	inline size_t batch_len_s() const {
		return m_a.shape(1);
	}

	inline size_t batch_len_t() const {
		return m_b.shape(1);
	}

	inline ValueVec operator()(const Index i, const Index j) const {
		ValueVec v;
		for (int k = 0; k < CellType::batch_size; k++) {
			v(k) = (m_a(k, i) == m_b(k, j)) ? m_eq : m_ne;
		}
		return v;
	}
};

template<typename Options, typename Algorithm>
class SolverImpl : public Solver<
	typename Algorithm::cell_type::value_type,
	typename Algorithm::cell_type::index_type> {
public:
	typedef typename Algorithm::cell_type CellType;
	typedef typename Algorithm::problem_type ProblemType;
	typedef typename CellType::value_type Value;
	typedef typename CellType::value_vec_type ValueVec;
	typedef typename CellType::index_type Index;

private:
	const std::shared_ptr<Options> m_options;
	Algorithm m_algorithm;

	template<typename Pairwise>
	inline xt::pytensor<Value, 1> _solve_for_score(
		const Pairwise &p_pairwise) const {

		ValueVec scores;

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_algorithm.solve(p_pairwise, p_pairwise.batch_len_s(), p_pairwise.batch_len_t());
			scores = m_algorithm.score(p_pairwise.len_s(), p_pairwise.len_t());
		}

		return scores;
	}

	template<typename Pairwise>
	inline py::tuple _solve_for_alignment(
		const Pairwise &p_pairwise) const {

		std::array<AlignmentRef<Index>, CellType::batch_size> alignments;

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_algorithm.solve(p_pairwise, p_pairwise.batch_len_s(), p_pairwise.batch_len_t());

			m_algorithm.template alignment<core::SharedPtrFactory<Alignment<Index>>>(
				p_pairwise.len_s(), p_pairwise.len_t(), alignments);
		}

		return to_tuple<AlignmentRef<Index>, CellType::batch_size>(alignments);
	}

	template<typename Pairwise>
	inline py::tuple _solve_for_alignment_iterator(
		const Pairwise &p_pairwise) const {

		std::array<AlignmentIteratorRef<Index>, CellType::batch_size> iterators;

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_algorithm.solve(p_pairwise, p_pairwise.batch_len_s(), p_pairwise.batch_len_t());

			size_t i = 0;
			for (auto iterator : m_algorithm.template alignment_iterator<
				core::SharedPtrFactory<Alignment<Index>>>(
				p_pairwise.len_s(), p_pairwise.len_t(), m_options->remove_dup())) {
				iterators.at(i++) = std::make_shared<AlignmentIteratorImpl<
					Index, typename Algorithm::locality_type>>(iterator);
			}
		}

		return to_tuple<AlignmentIteratorRef<Index>, CellType::batch_size>(iterators);
	}

	template<typename Pairwise>
	inline py::tuple _solve_for_solution(
		const Pairwise &p_pairwise) const {

		std::array<SolutionRef, CellType::batch_size> solutions;

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_algorithm.solve(p_pairwise, p_pairwise.batch_len_s(), p_pairwise.batch_len_t());

			std::array<std::shared_ptr<NativeSolution<CellType, ProblemType>>, CellType::batch_size> sol0;

			m_algorithm.template solution<
				core::SharedPtrFactory<Alignment<Index>>,
				core::SharedPtrFactory<NativeSolution<CellType, ProblemType>>>(
					p_pairwise.len_s(),
					p_pairwise.len_t(),
					sol0);

			for (int batch_i = 0; batch_i < m_algorithm.batch_size(); batch_i++) {
				solutions[batch_i] = std::make_shared<SolutionImpl<CellType, ProblemType>>(
					sol0[batch_i]);
			}
		}

		return to_tuple<SolutionRef, CellType::batch_size>(solutions);
	}

	template<typename Pairwise>
	inline py::tuple _solve_for_solution_iterator(
		const Pairwise &p_pairwise) const {

		std::array<SolutionIteratorRef, CellType::batch_size> iterators;

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_algorithm.solve(p_pairwise, p_pairwise.batch_len_s(), p_pairwise.batch_len_t());

			size_t i = 0;
			for (auto iterator : m_algorithm.template solution_iterator<
				core::SharedPtrFactory<Alignment<Index>>,
				core::SharedPtrFactory<NativeSolution<CellType, ProblemType>>>(
				p_pairwise.len_s(), p_pairwise.len_t(), m_options->remove_dup())) {

				iterators.at(i++) = std::make_shared<SolutionIteratorImpl<
					typename Algorithm::locality_type>>(iterator);
			}
		}

		return to_tuple<SolutionIteratorRef, CellType::batch_size>(iterators);
	}

public:
	template<typename... Args>
	inline SolverImpl(const Options &p_options, const Args... args) :
		m_options(p_options.clone()),
		m_algorithm(args...) {
	}

	virtual py::dict options() const override {
		return m_options->to_dict();
	}

	virtual int batch_size() const override {
		return m_algorithm.batch_size();
	}

	virtual xt::pytensor<Value, 1> solve_for_score(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_score(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual xt::pytensor<Value, 1> solve_indexed_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_score(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual xt::pytensor<Value, 1> solve_binary_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_score(
			binary_matrix_form<CellType>{p_a, p_b, p_eq, p_ne, p_length});
	}

	virtual py::tuple solve_for_alignment(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_alignment(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_for_alignment_iterator(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_alignment_iterator(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_alignment(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_alignment_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_alignment_iterator(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_binary_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_alignment(
			binary_matrix_form<CellType>{p_a, p_b, p_eq, p_ne, p_length});
	}

	virtual py::tuple solve_binary_for_alignment_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_alignment_iterator(
			binary_matrix_form<CellType>{p_a, p_b, p_eq, p_ne, p_length});
	}

	virtual py::tuple solve_for_solution(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_solution(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_for_solution_iterator(
		const xt::pytensor<Value, 3> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_solution_iterator(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_solution(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_solution_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<Value, 2> &p_similarity,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_solution_iterator(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_binary_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_solution(
			binary_matrix_form<CellType>{p_a, p_b, p_eq, p_ne, p_length});
	}

	virtual py::tuple solve_binary_for_solution_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const Value p_eq,
		const Value p_ne,
		const xt::pytensor<Index, 2> &p_length) const override {

		return _solve_for_solution_iterator(
			binary_matrix_form<CellType>{p_a, p_b, p_eq, p_ne, p_length});
	}

	const Algorithm &algorithm() const {
		return m_algorithm;
	}
};

template<typename Value, typename Index>
class SolverFactory {
public:
	virtual ~SolverFactory() {
	}

	virtual SolverRef<Value, Index> make(
		const size_t p_max_len_s,
		const size_t p_max_len_t) = 0;
};

template<typename Value, typename Index>
using SolverFactoryRef = std::shared_ptr<SolverFactory<Value, Index>>;

template<typename Value, typename Index, typename Generator>
class SolverFactoryImpl : public SolverFactory<Value, Index> {
	const Generator m_generator;

public:
	inline SolverFactoryImpl(
		const Generator &p_generator) :

		m_generator(p_generator) {
	}

	virtual SolverRef<Value, Index> make(
		const size_t p_max_len_s,
		const size_t p_max_len_t) {

		return m_generator(p_max_len_s, p_max_len_t);
	}
};

template<typename Options>
class MakeSolverImpl {
public:
	typedef typename Options::value_type Value;
	typedef typename Options::index_type Index;

	template<
		typename Algorithm,
		typename... Args>
	SolverFactoryRef<Value, Index> make(
		const Options &p_options,
		const Args... p_args) const {

		const auto gen = [=](
			const size_t p_max_len_s,
			const size_t p_max_len_t) {

			return std::make_shared<SolverImpl<Options, Algorithm>>(
				p_options,
				p_max_len_s,
				p_max_len_t,
				p_args...);
		};

		return std::make_shared<SolverFactoryImpl<Value, Index, decltype(gen)>>(gen);
	}
};

template<typename Options, typename MakeSolver>
struct FactoryCreator {
	const Options &m_options;
	const MakeSolver &m_make_solver;

    typedef typename Options::value_type Value;
    typedef typename Options::index_type Index;

	typedef core::cell_type<Value, Index, core::no_batch> cell_type_nobatch;
	typedef core::cell_type<Value, Index, core::machine_batch_size> cell_type_batched;

public:
	inline FactoryCreator(
		const Options &p_options,
		const MakeSolver &p_make_solver) :
		m_options(p_options),
		m_make_solver(p_make_solver) {
	}

	auto create_dtw_solver_factory() const {
		if (m_options.batch()) {
			return create_dtw_solver_factory_with_cell_type<cell_type_batched>();
		} else {
			return create_dtw_solver_factory_with_cell_type<cell_type_nobatch>();
		}
	}

	auto create_alignment_solver_factory() const {
		if (m_options.batch()) {
			return create_alignment_solver_factory_with_cell_type<cell_type_batched>();
		} else {
			return create_alignment_solver_factory_with_cell_type<cell_type_nobatch>();
		}
	}

private:
	template<typename CellType>
	auto create_dtw_solver_factory_with_cell_type() const {
		switch (m_options.direction()) {
			case enums::Direction::MAXIMIZE: {
				typedef core::problem_type<
					core::goal::one_optimal_alignment,
					core::direction::maximize> ProblemType;

				return m_make_solver.template make<
					core::DynamicTimeSolver<CellType, ProblemType>>(m_options);
			} break;

#if PYALIGN_FEATURES_MINIMIZE
			case enums::Direction::MINIMIZE: {
				typedef core::problem_type<
					core::goal::one_optimal_alignment,
					core::direction::minimize> ProblemType;

				return m_make_solver.template make<
					core::DynamicTimeSolver<CellType, ProblemType>>(m_options);
			} break;
#endif

			default: {
				throw std::invalid_argument("illegal direction");
			} break;
		}
	}

	template<
		typename CellType,
		typename ProblemType,
		template<typename, typename> class Locality,
		typename LocalityInitializers>
	auto resolve_gap_type(
		const LocalityInitializers &p_loc_initializers) const {

		const auto &gap_s = m_options.gap_costs().s();
		const auto &gap_t = m_options.gap_costs().t();

		if (gap_s.linear.has_value() && gap_t.linear.has_value()) {
			return m_make_solver.template make<
				core::LinearGapCostSolver<CellType, ProblemType, Locality>>(
					m_options,
					*gap_s.linear,
					*gap_t.linear,
					p_loc_initializers);
		} else if (gap_s.affine.has_value() && gap_t.affine.has_value()) {
			return m_make_solver.template make<
				core::AffineGapCostSolver<CellType, ProblemType, Locality>>(
					m_options,
					*gap_s.affine,
					*gap_t.affine,
					p_loc_initializers
				);
		} else {
			return m_make_solver.template make<
				core::GeneralGapCostSolver<CellType, ProblemType, Locality>>(
					m_options,
					*gap_s.general,
					*gap_t.general,
					p_loc_initializers
				);
		}
	}

	template<
		typename CellType,
		typename Goal,
		template<typename, typename> class Locality,
		typename LocalityInitializers>
	auto resolve_direction(
		const LocalityInitializers &p_loc_initializers) const {

		switch (m_options.direction()) {
			case enums::Direction::MAXIMIZE: {
				typedef core::problem_type<Goal, core::direction::maximize> ProblemType;

				return resolve_gap_type<
					CellType, ProblemType, Locality, LocalityInitializers>(
						p_loc_initializers);
			} break;

#if PYALIGN_FEATURES_MINIMIZE
			case enums::Direction::MINIMIZE: {
				typedef core::problem_type<Goal, core::direction::minimize> ProblemType;

				return resolve_gap_type<
					CellType, ProblemType, Locality, LocalityInitializers>(
						p_loc_initializers);
			} break;
#endif

			default: {
				throw std::invalid_argument("illegal direction");
			} break;
		}
	}

	template<typename CellType, typename Goal>
	auto resolve_locality() const {
		switch (m_options.locality()) {
			case enums::Locality::LOCAL: {
				return resolve_direction<CellType, Goal, core::Local>(
					core::LocalInitializers());
			} break;

			case enums::Locality::GLOBAL: {
				return resolve_direction<CellType, Goal, core::Global>(
					core::GlobalInitializers());
			} break;

			case enums::Locality::SEMIGLOBAL: {
				return resolve_direction<CellType, Goal, core::Semiglobal>(
					core::SemiglobalInitializers());
			} break;

			default: {
				throw std::invalid_argument("invalid locality");
			} break;
		}
	}

	template<typename CellType>
	auto create_alignment_solver_factory_with_cell_type() const {
		switch (m_options.count()) {
			case enums::Count::ONE: {

				switch (m_options.detail()) {
#if PYALIGN_FEATURES_SCORE_ONLY
					case enums::Detail::SCORE: {
						return resolve_locality<CellType, core::goal::optimal_score>();
					} break;
#endif

					case enums::Detail::ALIGNMENT:
					case enums::Detail::SOLUTION: {
						return resolve_locality<CellType, core::goal::one_optimal_alignment>();
					} break;

					default: {
						throw std::invalid_argument("invalid detail");
					} break;
				}
			} break;

#if PYALIGN_FEATURES_ALL_SOLUTIONS
			case enums::Count::ALL: {
				switch (m_options.detail()) {
					case enums::Detail::SCORE:
					case enums::Detail::ALIGNMENT:
					case enums::Detail::SOLUTION: {
						return resolve_locality<CellType, core::goal::all_optimal_alignments>();
					} break;

					default: {
						throw std::invalid_argument("invalid detail");
					} break;

				} break;

			} break;
#endif

			default: {
				throw std::invalid_argument("invalid count");
			} break;
		}
	}
};

template<typename Options, typename MakeSolver>
auto create_solver_factory(
	const Options &p_options,
	const MakeSolver &p_make_solver) {

	const auto creator = FactoryCreator<Options, MakeSolver>(
		p_options, p_make_solver);

	switch (p_options.type()) {
		case enums::Type::ALIGNMENT: {
			return creator.create_alignment_solver_factory();
		} break;

#if PYALIGN_FEATURES_DTW
		case enums::Type::DTW: {
			return creator.create_dtw_solver_factory();
		} break;
#endif

		default: {
			throw std::invalid_argument("illegal solver type");
		} break;
	}
}

template<typename Options>
SolverRef<typename Options::value_type, typename Options::index_type> create_solver(
	const size_t p_max_len_s,
	const size_t p_max_len_t,
	const Options &p_options) {

	const auto factory = create_solver_factory<Options, MakeSolverImpl<Options>>(
		p_options, MakeSolverImpl<Options>());

	return factory->make(p_max_len_s, p_max_len_t);
}

} // pyalign

#endif // __PYALIGN_FACTORY_H__
