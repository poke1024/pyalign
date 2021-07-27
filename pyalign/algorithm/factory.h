#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <xtensor-python/pytensor.hpp>

#include "solver.h"

namespace pyalign {

namespace py = pybind11;

inline xt::xtensor<float, 1> zero_gap_tensor(const size_t p_len) {
	xt::xtensor<float, 1> w;
	w.resize({p_len});
	w.fill(0);
	return w;
}

inline core::GapTensorFactory<float> to_gap_tensor_factory(const py::object &p_gap) {
	if (p_gap.is_none()) {
		return zero_gap_tensor;
	} else {
		auto f = p_gap.attr("costs").cast<std::function<xt::pytensor<float, 1>(size_t)>>();
		return [f] (const size_t n) {
			py::gil_scoped_acquire acquire;
			return f(n).cast<xt::xtensor<float, 1>>();
		};
	}
}

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
				affine = core::AffineCost<float>(
					affine_tuple[1].cast<float>(),
					affine_tuple[0].cast<float>()
				);

	        } else if (cost.contains("linear")) {
	            linear = cost["linear"].cast<float>();
	        } else {
	            general = to_gap_tensor_factory(p_gap);
	        }
	    }
	}

	std::optional<float> linear;
	std::optional<core::AffineCost<float>> affine;
	std::optional<core::GapTensorFactory<float>> general;
};

inline auto to_gap_cost_options(const py::object p_gap_cost) {
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

	return std::make_pair<GapCostOptions>(
		GapCostOptions(gap_s),
		GapCostOptions(gap_t)
	);
}

class GapCosts {
	const std::pair<GapCostOptions, GapCostOptions> m_options;

public:
	inline GapCosts(const py::object p_gap_cost) : m_options(
		to_gap_cost_options(p_gap_cost)) {
	}

	const GapCostOptions &s() const {
		return std::get<0>(m_options);
	}

	const GapCostOptions &t() const {
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

class Options {
public:
	enum struct Type {
		ALIGNMENT,
		DTW
	};

	enum struct Direction {
		MINIMIZE,
		MAXIMIZE
	};

private:
	const py::dict m_options;
	const Type m_type;
	const bool m_batch;
	const Direction m_direction;

public:
	inline Options(
		const py::dict &p_options) :

		m_options(p_options),
		m_type(p_options["solver"].cast<Type>()),
		m_batch(p_options["batch"].cast<bool>()),
		m_direction(p_options["direction"].cast<Direction>()) {
	}

	virtual ~Options() {
	}

	inline py::dict to_dict() {
		return m_options;
	}

	inline Type type() const {
		return m_type;
	}

	inline bool batch() const {
		return m_batch;
	}

	inline Direction direction() const {
		return m_direction;
	}
};

typedef std::shared_ptr<Options> OptionsRef;

class AlignmentOptions : public Options {
public:
	enum struct Detail {
		SCORE,
		ALIGNMENT,
		SOLUTION
	};

	enum struct Count {
		ONE,
		ALL
	};

	enum struct Locality {
		LOCAL,
		GLOBAL,
		SEMIGLOBAL
	};

private:
	const Detail m_detail;
	const Count m_count;
	const Locality m_locality;
	const GapCosts m_gap_costs;

public:
	inline AlignmentOptions(
		const py::dict &p_options) :

		Options(p_options),
		m_detail(p_options["codomain"].attr("detail").cast<Detail>()),
		m_count(p_options["codomain"].attr("count").cast<Count>()),
		m_locality(p_options.contains("locality") ?
			p_options["locality"].cast<Locality>() : Locality::LOCAL),
		m_gap_costs(p_options.contains("gap_cost") ?
			p_options["gap_cost"] : py::none().cast<py::object>()) {
	}

	inline Detail detail() const {
		return m_detail;
	}

	inline Count count() const {
		return m_count;
	}

	inline Locality locality() const {
		return m_locality;
	}

	inline const GapCosts &gap_costs() const {
		return m_gap_costs;
	}
};

typedef std::shared_ptr<AlignmentOptions> AlignmentOptionsRef;

OptionsRef create_options(const py::dict &p_options);

class Solver {
public:
	virtual inline ~Solver() {
	}

	virtual py::dict options() const = 0;

	virtual int batch_size() const = 0;

	virtual xt::pytensor<float, 1> solve_for_score(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual xt::pytensor<float, 1> solve_indexed_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_for_alignment(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_for_alignment_iterator(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_alignment_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_for_solution(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_for_solution_iterator(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;

	virtual py::tuple solve_indexed_for_solution_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const = 0;
};

typedef std::shared_ptr<Solver> SolverRef;

template<typename CellType>
struct matrix_form {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<float, 3> &m_similarity;
	const xt::pytensor<uint16_t, 2> &m_length;

	inline void check() const {
		check_batch_size(m_similarity.shape(2), CellType::batch_size);

		if (m_length.shape(0) != 2 || m_length.shape(1) != CellType::batch_size) {
			std::ostringstream err;
			err << "m_length has shape (" << m_length.shape(0) << ", " <<
				m_length.shape(1) << "), expected (2, " << CellType::batch_size << ")";
			throw std::invalid_argument(err.str());
		}
	}

	inline size_t batch_len_s() const {
		return m_similarity.shape(0);
	}

	inline size_t batch_len_t() const {
		return m_similarity.shape(1);
	}

	inline IndexVec len_s() const {
		return xt::view(m_length, 0, xt::all());
	}

	inline IndexVec len_t() const {
		return xt::view(m_length, 1, xt::all());
	}

	inline ValueVec operator()(const Index i, const Index j) const {
		ValueVec v = xt::view(m_similarity, i, j, xt::all());
		return v;
	}
};

template<typename CellType>
struct indexed_matrix_form {
	typedef typename CellType::index_type Index;
	typedef typename CellType::index_vec_type IndexVec;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<uint32_t, 2> &m_a;
	const xt::pytensor<uint32_t, 2> &m_b;
	const xt::pytensor<float, 2> &m_similarity;
	const xt::pytensor<uint16_t, 2> &m_length;

	inline void check() const {
		check_batch_size(m_a.shape(0), CellType::batch_size);
		check_batch_size(m_b.shape(0), CellType::batch_size);

		if (xt::amax(m_a)() >= m_similarity.shape(0)) {
			throw std::invalid_argument("out of bounds index in a");
		}
		if (xt::amax(m_b)() >= m_similarity.shape(1)) {
			throw std::invalid_argument("out of bounds index in b");
		}

		if (m_length.shape(0) != 2 || m_length.shape(1) != CellType::batch_size) {
			std::ostringstream err;
			err << "m_length has shape (" << m_length.shape(0) << ", " <<
				m_length.shape(1) << "), expected (2, " << CellType::batch_size << ")";
			throw std::invalid_argument(err.str());
		}
	}

	inline size_t batch_len_s() const {
		return m_a.shape(1);
	}

	inline size_t batch_len_t() const {
		return m_b.shape(1);
	}

	inline IndexVec len_s() const {
		return xt::view(m_length, 0, xt::all());
	}

	inline IndexVec len_t() const {
		return xt::view(m_length, 1, xt::all());
	}

	inline ValueVec operator()(const Index i, const Index j) const {
		ValueVec v;
		for (int k = 0; k < CellType::batch_size; k++) {
			v(k) = m_similarity(m_a(k, i), m_b(k, j));
		}
		return v;
	}
};

template<typename Algorithm>
class SolverImpl : public Solver {
public:
	typedef typename Algorithm::cell_type CellType;
	typedef typename Algorithm::problem_type ProblemType;
	typedef typename Algorithm::index_type Index;
	typedef typename CellType::value_vec_type ValueVec;

private:
	const OptionsRef m_options;
	Algorithm m_algorithm;

	template<typename Pairwise>
	inline xt::pytensor<float, 1> _solve_for_score(
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
				p_pairwise.len_s(), p_pairwise.len_t())) {
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
				p_pairwise.len_s(), p_pairwise.len_t())) {

				iterators.at(i++) = std::make_shared<SolutionIteratorImpl<
					typename Algorithm::locality_type>>(iterator);
			}
		}

		return to_tuple<SolutionIteratorRef, CellType::batch_size>(iterators);
	}

public:
	template<typename... Args>
	inline SolverImpl(const OptionsRef &p_options, const Args... args) :
		m_options(p_options),
		m_algorithm(args...) {
	}

	virtual py::dict options() const override {
		return m_options->to_dict();
	}

	virtual int batch_size() const override {
		return m_algorithm.batch_size();
	}

	virtual xt::pytensor<float, 1> solve_for_score(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_score(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual xt::pytensor<float, 1> solve_indexed_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_score(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_for_alignment(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_alignment(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_for_alignment_iterator(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_alignment_iterator(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_alignment(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_alignment_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_alignment_iterator(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_for_solution(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_solution(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_for_solution_iterator(
		const xt::pytensor<float, 3> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_solution_iterator(
			matrix_form<CellType>{p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_solution(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	virtual py::tuple solve_indexed_for_solution_iterator(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity,
		const xt::pytensor<uint16_t, 2> &p_length) const override {

		return _solve_for_solution_iterator(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity, p_length});
	}

	const Algorithm &algorithm() const {
		return m_algorithm;
	}
};

class SolverFactory {
public:
	virtual ~SolverFactory() {
	}

	virtual SolverRef make(
		const size_t p_max_len_s,
		const size_t p_max_len_t) = 0;
};

typedef std::shared_ptr<SolverFactory> SolverFactoryRef;

template<typename Generator>
class SolverFactoryImpl : public SolverFactory {
	const Generator m_generator;

public:
	inline SolverFactoryImpl(
		const Generator &p_generator) :

		m_generator(p_generator) {
	}

	virtual SolverRef make(
		const size_t p_max_len_s,
		const size_t p_max_len_t) {

		return m_generator(p_max_len_s, p_max_len_t);
	}
};

class MakeSolverImpl {
public:
	template<
		typename Algorithm,
		typename Options,
		typename... Args>
	SolverFactoryRef make(
		const Options &p_options,
		const Args... p_args) const {

		const auto gen = [=](
			const size_t p_max_len_s,
			const size_t p_max_len_t) {

			return std::make_shared<SolverImpl<Algorithm>>(
				p_options,
				p_max_len_s,
				p_max_len_t,
				p_args...);
		};

		return std::make_shared<SolverFactoryImpl<decltype(gen)>>(gen);
	}
};

template<typename CellType, typename MakeSolver>
struct FactoryCreation {
	static auto create_dtw_solver_factory(
		const OptionsRef &p_options,
		const MakeSolver &p_make_solver) {

		switch (p_options->direction()) {
			case Options::Direction::MAXIMIZE: {
				typedef core::problem_type<
					core::goal::one_optimal_alignment,
					core::direction::maximize> ProblemType;

				return p_make_solver.template make<
					core::DynamicTimeSolver<CellType, ProblemType>>(p_options);
			} break;

			case Options::Direction::MINIMIZE: {
				typedef core::problem_type<
					core::goal::one_optimal_alignment,
					core::direction::minimize> ProblemType;

				return p_make_solver.template make<
					core::DynamicTimeSolver<CellType, ProblemType>>(p_options);
			} break;

			default: {
				throw std::invalid_argument("illegal direction");
			} break;
		}
	}

	template<
		typename ProblemType,
		template<typename, typename> class Locality,
		typename LocalityInitializers>
	static auto resolve_gap_type(
		const AlignmentOptionsRef &p_options,
		const MakeSolver &p_make_solver,
		const LocalityInitializers &p_loc_initializers) {

		const auto &gap_s = p_options->gap_costs().s();
		const auto &gap_t = p_options->gap_costs().t();

		if (gap_s.linear.has_value() && gap_t.linear.has_value()) {
			return p_make_solver.template make<
				core::LinearGapCostSolver<CellType, ProblemType, Locality>>(
					p_options,
					*gap_s.linear,
					*gap_t.linear,
					p_loc_initializers);
		} else if (gap_s.affine.has_value() && gap_t.affine.has_value()) {
			return p_make_solver.template make<
				core::AffineGapCostSolver<CellType, ProblemType, Locality>>(
					p_options,
					*gap_s.affine,
					*gap_t.affine,
					p_loc_initializers
				);
		} else {
			return p_make_solver.template make<
				core::GeneralGapCostSolver<CellType, ProblemType, Locality>>(
					p_options,
					*gap_s.general,
					*gap_t.general,
					p_loc_initializers
				);
		}
	}

	template<
		typename Goal,
		template<typename, typename> class Locality,
		typename LocalityInitializers>
	static auto resolve_direction(
		const AlignmentOptionsRef &p_options,
		const MakeSolver &p_make_solver,
		const LocalityInitializers &p_loc_initializers) {

		switch (p_options->direction()) {
			case Options::Direction::MAXIMIZE: {
				typedef core::problem_type<Goal, core::direction::maximize> ProblemType;

				return FactoryCreation::resolve_gap_type<
					ProblemType, Locality, LocalityInitializers>(
						p_options, p_make_solver, p_loc_initializers);
			} break;

			case Options::Direction::MINIMIZE: {
				typedef core::problem_type<Goal, core::direction::minimize> ProblemType;

				return FactoryCreation::resolve_gap_type<
					ProblemType, Locality, LocalityInitializers>(
						p_options, p_make_solver, p_loc_initializers);
			} break;

			default: {
				throw std::invalid_argument("illegal direction");
			} break;
		}
	}

	template<typename Goal>
	static auto resolve_locality(
		const AlignmentOptionsRef &p_options,
		const MakeSolver &p_make_solver) {

		switch (p_options->locality()) {
			case AlignmentOptions::Locality::LOCAL: {
				return resolve_direction<Goal, core::Local>(
					p_options,
					p_make_solver,
					core::LocalInitializers());
			} break;

			case AlignmentOptions::Locality::GLOBAL: {
				return resolve_direction<Goal, core::Global>(
					p_options,
					p_make_solver,
					core::GlobalInitializers());
			} break;

			case AlignmentOptions::Locality::SEMIGLOBAL: {
				return resolve_direction<Goal, core::Semiglobal>(
					p_options,
					p_make_solver,
					core::SemiglobalInitializers());
			} break;

			default: {
				throw std::invalid_argument("invalid locality");
			} break;
		}
	}

	static auto create_alignment_solver_factory(
		const AlignmentOptionsRef &p_options,
		const MakeSolver &p_make_solver) {

		switch (p_options->count()) {
			case AlignmentOptions::Count::ONE: {

				switch (p_options->detail()) {
					case AlignmentOptions::Detail::SCORE: {
						return FactoryCreation::
							template resolve_locality<core::goal::optimal_score>(
								p_options,
								p_make_solver);
					} break;

					case AlignmentOptions::Detail::ALIGNMENT:
					case AlignmentOptions::Detail::SOLUTION: {
						return FactoryCreation::
							template resolve_locality<core::goal::one_optimal_alignment>(
								p_options,
								p_make_solver);
					} break;

					default: {
						throw std::invalid_argument("invalid detail");
					} break;
				}
			} break;

			case AlignmentOptions::Count::ALL: {

				switch (p_options->detail()) {

					case AlignmentOptions::Detail::SCORE:
					case AlignmentOptions::Detail::ALIGNMENT:
					case AlignmentOptions::Detail::SOLUTION: {
						return FactoryCreation::
							template resolve_locality<core::goal::all_optimal_alignments>(
								p_options,
								p_make_solver);
					} break;

					default: {
						throw std::invalid_argument("invalid detail");
					} break;

				} break;

			} break;

			default: {
				throw std::invalid_argument("invalid count");
			} break;
		}
	}
};

template<typename MakeSolver, typename Value, typename Index>
auto create_solver_factory(
	const OptionsRef &p_options,
	const MakeSolver &p_make_solver) {

	typedef core::cell_type<Value, Index, core::no_batch> cell_type_nobatch;
	typedef core::cell_type<Value, Index, core::machine_batch_size> cell_type_batched;

	switch (p_options->type()) {
		case Options::Type::ALIGNMENT: {
			if (p_options->batch()) {
				return FactoryCreation<cell_type_batched, MakeSolver>::
					create_alignment_solver_factory(
						std::dynamic_pointer_cast<AlignmentOptions>(
							p_options), p_make_solver);
			} else {
				return FactoryCreation<cell_type_nobatch, MakeSolver>::
					create_alignment_solver_factory(
						std::dynamic_pointer_cast<AlignmentOptions>(
							p_options), p_make_solver);
			}
		} break;

		case Options::Type::DTW: {
			if (p_options->batch()) {
				return FactoryCreation<cell_type_batched, MakeSolver>::
					create_dtw_solver_factory(p_options, p_make_solver);
			} else {
				return FactoryCreation<cell_type_nobatch, MakeSolver>::
					create_dtw_solver_factory(p_options, p_make_solver);
			}
		} break;

		default: {
			throw std::invalid_argument("illegal solver type");
		} break;
	}
}

} // pyalign
