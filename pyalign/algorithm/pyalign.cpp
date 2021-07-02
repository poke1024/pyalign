#define FORCE_IMPORT_ARRAY

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <xtensor-python/pytensor.hpp>

#include "solver.h"

namespace py = pybind11;

typedef int16_t cell_index_t;
typedef pyalign::cell_type<float, cell_index_t, pyalign::no_batch> cell_type_nobatch;
typedef pyalign::cell_type<float, cell_index_t, pyalign::machine_batch_size> cell_type_batched;

class Alignment {
public:
	typedef cell_index_t Index;

private:
	xt::pytensor<Index, 1> m_s_to_t;
	xt::pytensor<Index, 1> m_t_to_s;
	float m_score;

public:
	inline void resize(const size_t len_s, const size_t len_t) {
		m_s_to_t.resize({static_cast<ssize_t>(len_s)});
		m_s_to_t.fill(-1);

		m_t_to_s.resize({static_cast<ssize_t>(len_t)});
		m_t_to_s.fill(-1);
	}

	inline void add_edge(const size_t u, const size_t v) {
		m_s_to_t[u] = v;
		m_t_to_s[v] = u;
	}

	inline void set_score(const float p_score) {
		m_score = p_score;
	}

	inline float score() const {
		return m_score;
	}

	inline const xt::pytensor<Index, 1> &s_to_t() const {
		return m_s_to_t;
	}

	inline const xt::pytensor<Index, 1> &t_to_s() const {
		return m_t_to_s;
	}
};

typedef std::shared_ptr<Alignment> AlignmentRef;

typedef pyalign::AlgorithmMetaData Algorithm;
typedef pyalign::AlgorithmMetaDataRef AlgorithmRef;

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
	virtual xt::pytensor<Index, 2> path() const = 0;
	virtual float score() const = 0;
	virtual AlgorithmRef algorithm() const = 0;
	virtual AlignmentRef alignment() const = 0;
};

typedef std::shared_ptr<Solution> SolutionRef;

template<typename CellType, typename ProblemType>
class SolutionImpl : public Solution {
public:
	typedef typename CellType::value_type Value;
	typedef typename CellType::index_type Index;

private:
	const pyalign::SolutionRef<CellType, ProblemType> m_solution;
	AlignmentRef m_alignment;
	const int m_batch_index;

public:
	inline SolutionImpl(
		const pyalign::SolutionRef<CellType, ProblemType> p_solution,
		const AlignmentRef &p_alignment,
		const int p_batch_index) :

		m_solution(p_solution),
		m_alignment(p_alignment),
		m_batch_index(p_batch_index) {
	}

	virtual bool traceback_has_max_degree_1() const override {
		return m_solution->has_degree_1_traceback();
	}

	virtual xt::pytensor<Value, 3> values() const override {
		return m_solution->values(m_batch_index);
	}

	virtual xt::pytensor<Index, 4> traceback_as_matrix() const override {
		return m_solution->traceback_as_matrix(m_batch_index);
	}

	virtual py::list traceback_as_edges() const override {
		const auto edges = m_solution->traceback_as_edges(m_batch_index);

		py::list py_edges;
		for (const auto &layer_edges : edges) {
			py_edges.append(xt::pytensor<Index, 3>(layer_edges));
		}

		return py_edges;
	}

	virtual xt::pytensor<Index, 2> path() const override {
		return m_solution->path();
	}

	virtual float score() const override {
		return m_solution->score();
	}

	virtual AlgorithmRef algorithm() const override {
		return m_solution->algorithm();
	}

	virtual AlignmentRef alignment() const override {
		return m_alignment;
	}
};

class Solver {
public:
	virtual inline ~Solver() {
	}

	virtual const py::dict &options() const = 0;

	virtual int batch_size() const = 0;

	virtual xt::pytensor<float, 1> solve_for_score(
		const xt::pytensor<float, 3> &p_similarity) const = 0;

	virtual xt::pytensor<float, 1> solve_indexed_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity) const = 0;

	virtual py::tuple solve_for_alignment(
		const xt::pytensor<float, 3> &p_similarity) const = 0;

	virtual py::tuple solve_indexed_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity) const = 0;

	virtual py::tuple solve_for_solution(
		const xt::pytensor<float, 3> &p_similarity) const = 0;

	virtual py::tuple solve_indexed_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity) const = 0;
};

typedef std::shared_ptr<Solver> SolverRef;


inline void check_batch_size(const size_t given, const size_t batch_size) {
	if (given != batch_size) {
		std::ostringstream err;
		err << "problem of batch size " << given <<
			" given to solver that was configured to batch size " << batch_size;
		throw std::invalid_argument(err.str());
	}
}

template<typename Array, std::size_t... T>
py::tuple make_obj_tuple(const Array& array, std::index_sequence<T...>) {
    return py::make_tuple(array[T]...);
}

template<typename CellType>
struct matrix_form {
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<float, 3> &m_similarity;

	inline void check() const {
		check_batch_size(m_similarity.shape(2), CellType::batch_size);
	}

	inline size_t len_s() const {
		return m_similarity.shape(0);
	}

	inline size_t len_t() const {
		return m_similarity.shape(1);
	}

	inline ValueVec operator()(const Index i, const Index j) const {
		ValueVec v = xt::view(m_similarity, i, j, xt::all());
		return v;
	}
};

template<typename CellType>
struct indexed_matrix_form {
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_vec_type ValueVec;

	const xt::pytensor<uint32_t, 2> &m_a;
	const xt::pytensor<uint32_t, 2> &m_b;
	const xt::pytensor<float, 2> &m_similarity;

	inline void check() const {
		check_batch_size(m_a.shape(0), CellType::batch_size);
		check_batch_size(m_b.shape(0), CellType::batch_size);

		if (xt::amax(m_a)() >= m_similarity.shape(0)) {
			throw std::invalid_argument("out of bounds index in a");
		}
		if (xt::amax(m_b)() >= m_similarity.shape(1)) {
			throw std::invalid_argument("out of bounds index in b");
		}
	}

	inline size_t len_s() const {
		return m_a.shape(1);
	}

	inline size_t len_t() const {
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

template<typename CellType, typename ProblemType, typename S>
class SolverImpl : public Solver {
private:
	const py::dict m_options;
	S m_solver;

	template<typename Pairwise>
	inline xt::pytensor<float, 1> _solve_for_score(
		const Pairwise &p_pairwise) const {

		ValueVec scores;

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_solver.solve(p_pairwise, p_pairwise.len_s(), p_pairwise.len_t());
			scores = m_solver.score(p_pairwise.len_s(), p_pairwise.len_t());
		}

		return scores;
	}

	template<typename Pairwise>
	inline py::tuple _solve_for_alignment(
		const Pairwise &p_pairwise) const {

		AlignmentRef alignments[CellType::batch_size];

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_solver.solve(p_pairwise, p_pairwise.len_s(), p_pairwise.len_t());

			for (int batch_i = 0; batch_i < m_solver.batch_size(); batch_i++) {
				AlignmentRef alignment = std::make_shared<Alignment>();
				const float score = m_solver.alignment(
					p_pairwise.len_s(), p_pairwise.len_t(),
					*alignment.get(), batch_i);
				alignment->set_score(score);
				alignments[batch_i] = alignment;
			}
		}

		py::object py_alignments[CellType::batch_size];

		for (int batch_i = 0; batch_i < m_solver.batch_size(); batch_i++) {
			py_alignments[batch_i] = py::cast(alignments[batch_i]);
		}

		return make_obj_tuple(
			py_alignments,
			std::make_index_sequence<CellType::batch_size>());
	}

	template<typename Pairwise>
	inline py::tuple _solve_for_solution(
		const Pairwise &p_pairwise) const {

		SolutionRef solutions[CellType::batch_size];

		{
			py::gil_scoped_release release;
			p_pairwise.check();
			m_solver.solve(p_pairwise, p_pairwise.len_s(), p_pairwise.len_t());

			for (int batch_i = 0; batch_i < m_solver.batch_size(); batch_i++) {
				const auto alignment = std::make_shared<Alignment>();
				SolutionRef solution = std::make_shared<SolutionImpl<CellType, ProblemType>>(
					m_solver.solution(p_pairwise.len_s(), p_pairwise.len_t(), *alignment.get(), batch_i),
					alignment,
					batch_i);
				solutions[batch_i] = solution;
			}
		}

		py::object py_solutions[CellType::batch_size];

		for (int batch_i = 0; batch_i < m_solver.batch_size(); batch_i++) {
			py_solutions[batch_i] = py::cast(solutions[batch_i]);
		}

		return make_obj_tuple(
			py_solutions,
			std::make_index_sequence<CellType::batch_size>());
	}

public:
	typedef typename CellType::index_type Index;
	typedef typename CellType::value_vec_type ValueVec;

	template<typename... Args>
	inline SolverImpl(const py::dict &p_options, const Args&... args) :
		m_options(p_options),
		m_solver(args...) {
	}

	virtual const py::dict &options() const override {
		return m_options;
	}

	virtual int batch_size() const override {
		return m_solver.batch_size();
	}

	virtual xt::pytensor<float, 1> solve_for_score(
		const xt::pytensor<float, 3> &p_similarity) const override {

		return _solve_for_score(
			matrix_form<CellType>{p_similarity});
	}

	virtual xt::pytensor<float, 1> solve_indexed_for_score(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity) const override {

		return _solve_for_score(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity});
	}

	virtual py::tuple solve_for_alignment(
		const xt::pytensor<float, 3> &p_similarity) const override {

		return _solve_for_alignment(
			matrix_form<CellType>{p_similarity});
	}

	virtual py::tuple solve_indexed_for_alignment(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity) const override {

		return _solve_for_alignment(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity});
	}

	virtual py::tuple solve_for_solution(
		const xt::pytensor<float, 3> &p_similarity) const override {

		return _solve_for_solution(
			matrix_form<CellType>{p_similarity});
	}

	virtual py::tuple solve_indexed_for_solution(
		const xt::pytensor<uint32_t, 2> &p_a,
		const xt::pytensor<uint32_t, 2> &p_b,
		const xt::pytensor<float, 2> &p_similarity) const override {

		return _solve_for_solution(
			indexed_matrix_form<CellType>{p_a, p_b, p_similarity});
	}
};

inline xt::pytensor<float, 1> zero_gap_tensor(const size_t p_len) {
	xt::pytensor<float, 1> w;
	w.resize({static_cast<ssize_t>(p_len)});
	w.fill(0);
	return w;
}

inline pyalign::GapTensorFactory<float> to_gap_tensor_factory(const py::object &p_gap) {
	if (p_gap.is_none()) {
		return zero_gap_tensor;
	} else {
		return p_gap.attr("costs").cast<pyalign::GapTensorFactory<float>>();
	}
}

struct GapCostSpecialCases {
	inline GapCostSpecialCases(const py::object &p_gap) {
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
				affine = pyalign::AffineCost<float>(
					affine_tuple[1].cast<float>(),
					affine_tuple[0].cast<float>()
				);

	        } else if (cost.contains("linear")) {
	            linear = cost["linear"].cast<float>();
	        }
	    }
	}

	std::optional<float> linear;
	std::optional<pyalign::AffineCost<float>> affine;
};

template<typename CellType>
struct AlignmentSolverFactory {

	template<template<typename, typename, template<typename, typename> class Locality> class AlignmentSolver,
		typename Goal, template<typename, typename> class Locality, typename... Args>
	static SolverRef resolve_direction(
		const py::dict &p_options,
		const Args&... args) {

		const std::string direction = p_options["direction"].cast<std::string>();

		if (direction == "maximize") {
			typedef pyalign::problem_type<Goal, pyalign::direction::maximize> ProblemType;
			return std::make_shared<SolverImpl<CellType, ProblemType,
				AlignmentSolver<CellType, ProblemType, Locality>>>(
					p_options, args...);
		} else if (direction == "minimize") {
			typedef pyalign::problem_type<Goal, pyalign::direction::minimize> ProblemType;
			return std::make_shared<SolverImpl<CellType, ProblemType,
				AlignmentSolver<CellType, ProblemType, Locality>>>(
					p_options, args...);
		} else {
			throw std::invalid_argument(direction);
		}
	}

	template<typename Goal, template<typename, typename> class Locality, typename LocalityInitializers>
	static SolverRef resolve_gap_type(
		const py::dict &p_options,
		const LocalityInitializers &p_loc_initializers,
		const size_t p_max_len_s,
		const size_t p_max_len_t) {

		const py::object gap_cost = p_options.contains("gap_cost") ?
			p_options["gap_cost"] : py::none().cast<py::object>();

		py::object gap_s = py::none();
		py::object gap_t = py::none();

		if (py::isinstance<py::dict>(gap_cost)) {
			const py::dict gap_cost_dict = gap_cost.cast<py::dict>();

			if (gap_cost_dict.contains("s")) {
				gap_s = gap_cost_dict["s"];
			}
			if (gap_cost_dict.contains("t")) {
				gap_t = gap_cost_dict["t"];
			}
		} else {
			gap_s = gap_cost;
			gap_t = gap_cost;
		}

		const GapCostSpecialCases x_gap_s(gap_s);
		const GapCostSpecialCases x_gap_t(gap_t);

		if (x_gap_s.linear.has_value() && x_gap_t.linear.has_value()) {
			return AlignmentSolverFactory::resolve_direction<pyalign::LinearGapCostSolver, Goal, Locality>(
				p_options,
				x_gap_s.linear.value(),
				x_gap_t.linear.value(),
				p_max_len_s,
				p_max_len_t,
				p_loc_initializers
			);
		} else if (x_gap_s.affine.has_value() && x_gap_t.affine.has_value()) {
			return AlignmentSolverFactory::resolve_direction<pyalign::AffineGapCostSolver, Goal, Locality>(
				p_options,
				x_gap_s.affine.value(),
				x_gap_t.affine.value(),
				p_max_len_s,
				p_max_len_t,
				p_loc_initializers
			);
		} else {
			return AlignmentSolverFactory::resolve_direction<pyalign::GeneralGapCostSolver, Goal, Locality>(
				p_options,
				to_gap_tensor_factory(gap_s),
				to_gap_tensor_factory(gap_t),
				p_max_len_s,
				p_max_len_t,
				p_loc_initializers
			);
		}
	}

	template<typename Goal>
	static SolverRef resolve_locality(
		const py::dict &p_options,
		const size_t p_max_len_s,
		const size_t p_max_len_t) {

		const std::string locality_name = p_options.contains("locality") ?
			p_options["locality"].cast<std::string>() : "local";

		if (locality_name == "local") {
			return resolve_gap_type<Goal, pyalign::Local>(
				p_options,
				pyalign::LocalInitializers(),
				p_max_len_s,
				p_max_len_t);

		} else if (locality_name == "global") {

			return resolve_gap_type<Goal, pyalign::Global>(
				p_options,
				pyalign::GlobalInitializers(),
				p_max_len_s,
				p_max_len_t);

		} else if (locality_name == "semiglobal") {

			return resolve_gap_type<Goal, pyalign::Semiglobal>(
				p_options,
				pyalign::SemiglobalInitializers(),
				p_max_len_s,
				p_max_len_t);

		} else {

			throw std::invalid_argument(locality_name);
		}
	}
};

template<typename CellType>
SolverRef create_alignment_solver(
	const size_t p_max_len_s,
	const size_t p_max_len_t,
	const py::dict &p_options) {

	const auto goal = p_options["goal"];
	const auto detail = goal.attr("detail").cast<std::string>();
	const auto count = goal.attr("count").cast<std::string>();

	if (count == "one") {
		if (detail == "score") {
			return AlignmentSolverFactory<CellType>::template resolve_locality<pyalign::goal::optimal_score>(
				p_options,
				p_max_len_s,
				p_max_len_t);
		} else if (detail == "alignment" || detail == "solution") {
			return AlignmentSolverFactory<CellType>::template resolve_locality<pyalign::goal::one_optimal_alignment>(
				p_options,
				p_max_len_s,
				p_max_len_t);
		} else {
			throw std::invalid_argument(detail);
		}
	} else if (count == "all") {
		if (detail == "alignment" || detail == "solution") {
			return AlignmentSolverFactory<CellType>::template resolve_locality<pyalign::goal::all_optimal_alignments>(
				p_options,
				p_max_len_s,
				p_max_len_t);
		} else {
			throw std::invalid_argument(detail);
		}
	} else {
		throw std::invalid_argument(count);
	}
}

template<typename CellType>
SolverRef create_dtw_solver(
	const size_t p_max_len_s,
	const size_t p_max_len_t,
	const py::dict &p_options) {

	const std::string direction = p_options["direction"].cast<std::string>();

	if (direction == "maximize") {
		typedef pyalign::problem_type<
			pyalign::goal::one_optimal_alignment,
			pyalign::direction::maximize> ProblemType;
		return std::make_shared<SolverImpl<CellType, ProblemType,
			pyalign::DynamicTimeSolver<CellType, ProblemType>>>(
				p_options,
				p_max_len_s,
				p_max_len_t);
	} else if (direction == "minimize") {
		typedef pyalign::problem_type<
			pyalign::goal::one_optimal_alignment,
			pyalign::direction::minimize> ProblemType;
		return std::make_shared<SolverImpl<CellType, ProblemType,
			pyalign::DynamicTimeSolver<CellType, ProblemType>>>(
				p_options,
				p_max_len_s,
				p_max_len_t);
	} else {
		throw std::invalid_argument(direction);
	}
}

SolverRef create_solver(
	const size_t p_max_len_s,
	const size_t p_max_len_t,
	const py::dict &p_options) {

	const std::string solver = p_options["solver"].cast<py::str>();
	const bool batch = p_options["batch"].cast<bool>();

	if (solver == "alignment") {
		if (batch) {
			return create_alignment_solver<cell_type_batched>(
				p_max_len_s, p_max_len_t, p_options);
		} else {
			return create_alignment_solver<cell_type_nobatch>(
				p_max_len_s, p_max_len_t, p_options);
		}
	} else if (solver == "dtw") {
		if (batch) {
			return create_dtw_solver<cell_type_batched>(
				p_max_len_s, p_max_len_t, p_options);
		} else {
			return create_dtw_solver<cell_type_nobatch>(
				p_max_len_s, p_max_len_t, p_options);
		}
	} else {
		throw std::invalid_argument(solver);
	}
}

PYBIND11_MODULE(algorithm, m) {
	xt::import_numpy();

	m.def("create_solver", &create_solver);

	py::class_<Solver, SolverRef> solver(m, "Solver");
	solver.def_property_readonly("options", &Solver::options);
	solver.def_property_readonly("batch_size", &Solver::batch_size);
	solver.def("solve_for_score", &Solver::solve_for_score);
	solver.def("solve_indexed_for_score", &Solver::solve_indexed_for_score);
	solver.def("solve_for_alignment", &Solver::solve_for_alignment);
	solver.def("solve_indexed_for_alignment", &Solver::solve_indexed_for_alignment);
	solver.def("solve_for_solution", &Solver::solve_for_solution);
	solver.def("solve_indexed_for_solution", &Solver::solve_indexed_for_solution);

	py::class_<Alignment, AlignmentRef> alignment(m, "Alignment");
	alignment.def_property_readonly("score", &Alignment::score);
	alignment.def_property_readonly("s_to_t", &Alignment::s_to_t);
	alignment.def_property_readonly("t_to_s", &Alignment::t_to_s);

	py::class_<Solution, SolutionRef> solution(m, "Solution");
	solution.def_property_readonly("values", &Solution::values);
	solution.def_property_readonly("traceback_has_max_degree_1", &Solution::traceback_has_max_degree_1);
	solution.def_property_readonly("traceback_as_matrix", &Solution::traceback_as_matrix);
	solution.def_property_readonly("traceback_as_edges", &Solution::traceback_as_edges);
	solution.def_property_readonly("path", &Solution::path);
	solution.def_property_readonly("score", &Solution::score);
	solution.def_property_readonly("alignment", &Solution::alignment);
	solution.def_property_readonly("algorithm", &Solution::algorithm);

	py::class_<Algorithm, AlgorithmRef> algorithm(m, "Algorithm");
	algorithm.def_property_readonly("name", &Algorithm::name);
	algorithm.def_property_readonly("runtime", &Algorithm::runtime);
	algorithm.def_property_readonly("memory", &Algorithm::memory);
}
