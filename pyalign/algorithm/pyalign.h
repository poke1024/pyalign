#ifndef __PYALIGN_PYALIGN_H__
#define __PYALIGN_PYALIGN_H__ 1

#include "pyalign/algorithm/common.h"
#include "pyalign/algorithm/factory.h"

namespace py = pybind11;

namespace pyalign {

template<typename Value, typename Index>
void register_solver(py::module_ &m) {
	m.def("create_solver", &create_solver<Value, Index>);

	py::class_<Solver<Value, Index>, SolverRef<Value, Index>> solver(m, "Solver");
	solver.def_property_readonly("options", &Solver<Value, Index>::options);
	solver.def_property_readonly("batch_size", &Solver<Value, Index>::batch_size);
	solver.def("solve_for_score", &Solver<Value, Index>::solve_for_score);
	solver.def("solve_indexed_for_score", &Solver<Value, Index>::solve_indexed_for_score);
	solver.def("solve_for_alignment", &Solver<Value, Index>::solve_for_alignment);
	solver.def("solve_for_alignment_iterator", &Solver<Value, Index>::solve_for_alignment_iterator);
	solver.def("solve_indexed_for_alignment", &Solver<Value, Index>::solve_indexed_for_alignment);
	solver.def("solve_indexed_for_alignment_iterator", &Solver<Value, Index>::solve_indexed_for_alignment_iterator);
	solver.def("solve_for_solution", &Solver<Value, Index>::solve_for_solution);
	solver.def("solve_for_solution_iterator", &Solver<Value, Index>::solve_for_solution_iterator);
	solver.def("solve_indexed_for_solution", &Solver<Value, Index>::solve_indexed_for_solution);
	solver.def("solve_indexed_for_solution_iterator", &Solver<Value, Index>::solve_indexed_for_solution_iterator);

	py::class_<Alignment<Index>, AlignmentRef<Index>> alignment(m, "Alignment");
	alignment.def_property_readonly("score", &Alignment<Index>::score);
	alignment.def_property_readonly("s_to_t", &Alignment<Index>::s_to_t);
	alignment.def_property_readonly("t_to_s", &Alignment<Index>::t_to_s);

	py::class_<AlignmentIterator<Index>, AlignmentIteratorRef<Index>> alignment_iterator(m, "AlignmentIterator");
	alignment_iterator.def("next", &AlignmentIterator<Index>::next);

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
	py::class_<AlignmentOptions<Value>, Options, AlignmentOptionsRef<Value>>
		alignment_options(m, "AlignmentOptions");
	m.def("create_options", &create_options<Value>);

	py::enum_<Options::Type>(m, "Type")
        .value("ALIGNMENT", Options::Type::ALIGNMENT)
        .value("DTW", Options::Type::DTW);

	py::enum_<Options::Direction>(m, "Direction")
        .value("MINIMIZE", Options::Direction::MINIMIZE)
        .value("MAXIMIZE", Options::Direction::MAXIMIZE);

	py::enum_<typename AlignmentOptions<Value>::Detail>(m, "Detail")
        .value("SCORE", AlignmentOptions<Value>::Detail::SCORE)
        .value("ALIGNMENT", AlignmentOptions<Value>::Detail::ALIGNMENT)
        .value("SOLUTION", AlignmentOptions<Value>::Detail::SOLUTION);

	py::enum_<typename AlignmentOptions<Value>::Count>(m, "Count")
        .value("ONE", AlignmentOptions<Value>::Count::ONE)
        .value("ALL", AlignmentOptions<Value>::Count::ALL);

	py::enum_<typename AlignmentOptions<Value>::Locality>(m, "Locality")
        .value("LOCAL", AlignmentOptions<Value>::Locality::LOCAL)
        .value("GLOBAL", AlignmentOptions<Value>::Locality::GLOBAL)
        .value("SEMIGLOBAL", AlignmentOptions<Value>::Locality::SEMIGLOBAL);
}

} // pyalign

#endif // __PYALIGN_PYALIGN_H__
