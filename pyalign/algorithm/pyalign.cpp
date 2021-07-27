#define FORCE_IMPORT_ARRAY

#include "solver.h"
#include "factory.h"

namespace py = pybind11;

using namespace pyalign;

typedef float py_value;
typedef int16_t py_index;

SolverRef create_solver(
	const size_t p_max_len_s,
	const size_t p_max_len_t,
	const OptionsRef &p_options) {

	const auto factory = create_solver_factory<MakeSolverImpl, py_value, py_index>(
		p_options, MakeSolverImpl());

	return factory->make(p_max_len_s, p_max_len_t);
}

OptionsRef create_options(const py::dict &p_options) {
	if (p_options["solver"].cast<Options::Type>() == Options::Type::ALIGNMENT) {
		return std::make_shared<AlignmentOptions>(p_options);
	} else {
		return std::make_shared<Options>(p_options);
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
	solver.def("solve_for_alignment_iterator", &Solver::solve_for_alignment_iterator);
	solver.def("solve_indexed_for_alignment", &Solver::solve_indexed_for_alignment);
	solver.def("solve_indexed_for_alignment_iterator", &Solver::solve_indexed_for_alignment_iterator);
	solver.def("solve_for_solution", &Solver::solve_for_solution);
	solver.def("solve_for_solution_iterator", &Solver::solve_for_solution_iterator);
	solver.def("solve_indexed_for_solution", &Solver::solve_indexed_for_solution);
	solver.def("solve_indexed_for_solution_iterator", &Solver::solve_indexed_for_solution_iterator);

	py::class_<Alignment<py_index>, AlignmentRef<py_index>> alignment(m, "Alignment");
	alignment.def_property_readonly("score", &Alignment<py_index>::score);
	alignment.def_property_readonly("s_to_t", &Alignment<py_index>::s_to_t);
	alignment.def_property_readonly("t_to_s", &Alignment<py_index>::t_to_s);

	py::class_<AlignmentIterator<py_index>, AlignmentIteratorRef<py_index>> alignment_iterator(m, "AlignmentIterator");
	alignment_iterator.def("next", &AlignmentIterator<py_index>::next);

	py::class_<Solution, SolutionRef> solution(m, "Solution");
	solution.def_property_readonly("values", &Solution::values);
	solution.def_property_readonly("traceback_has_max_degree_1", &Solution::traceback_has_max_degree_1);
	solution.def_property_readonly("traceback_as_matrix", &Solution::traceback_as_matrix);
	solution.def_property_readonly("traceback_as_edges", &Solution::traceback_as_edges);
	solution.def_property_readonly("path", &Solution::path);
	solution.def_property_readonly("score", &Solution::score);
	solution.def_property_readonly("alignment", &Solution::alignment);
	solution.def_property_readonly("algorithm", &Solution::algorithm);

	py::class_<SolutionIterator, SolutionIteratorRef> solution_iterator(m, "SolutionIterator");
	solution_iterator.def("next", &SolutionIterator::next);

	py::class_<Algorithm, AlgorithmRef> algorithm(m, "Algorithm");
	algorithm.def_property_readonly("name", &Algorithm::name);
	algorithm.def_property_readonly("runtime", &Algorithm::runtime);
	algorithm.def_property_readonly("memory", &Algorithm::memory);

	py::class_<Options, OptionsRef> options(m, "Options");
	py::class_<AlignmentOptions, Options, AlignmentOptionsRef>
		alignment_options(m, "AlignmentOptions");
	m.def("create_options", &create_options);

	py::enum_<Options::Type>(m, "Type")
        .value("ALIGNMENT", Options::Type::ALIGNMENT)
        .value("DTW", Options::Type::DTW);

	py::enum_<Options::Direction>(m, "Direction")
        .value("MINIMIZE", Options::Direction::MINIMIZE)
        .value("MAXIMIZE", Options::Direction::MAXIMIZE);

	py::enum_<AlignmentOptions::Detail>(m, "Detail")
        .value("SCORE", AlignmentOptions::Detail::SCORE)
        .value("ALIGNMENT", AlignmentOptions::Detail::ALIGNMENT)
        .value("SOLUTION", AlignmentOptions::Detail::SOLUTION);

	py::enum_<AlignmentOptions::Count>(m, "Count")
        .value("ONE", AlignmentOptions::Count::ONE)
        .value("ALL", AlignmentOptions::Count::ALL);

	py::enum_<AlignmentOptions::Locality>(m, "Locality")
        .value("LOCAL", AlignmentOptions::Locality::LOCAL)
        .value("GLOBAL", AlignmentOptions::Locality::GLOBAL)
        .value("SEMIGLOBAL", AlignmentOptions::Locality::SEMIGLOBAL);
}
