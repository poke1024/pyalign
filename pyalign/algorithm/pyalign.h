#ifndef __PYALIGN_PYALIGN_H__
#define __PYALIGN_PYALIGN_H__ 1

#include "pyalign/algorithm/common.h"
#include "pyalign/algorithm/factory.h"
#include "pyalign/algorithm/options.h"

namespace py = pybind11;

namespace pyalign {

template<typename Options>
std::shared_ptr<Options> create_options(const py::dict &p_options) {
	return std::make_shared<Options>(p_options);
}

template<typename Options>
void register_solver(py::module_ &m) {
	typedef typename Options::value_type Value;
	typedef typename Options::index_type Index;

	m.def("create_solver", &create_solver<Options>);

	py::class_<Solver<Value, Index>, SolverRef<Value, Index>> solver(m, "Solver", py::module_local());
	solver.def_property_readonly("options", &Solver<Value, Index>::options);
	solver.def_property_readonly("batch_size", &Solver<Value, Index>::batch_size);
	solver.def("solve_for_score", &Solver<Value, Index>::solve_for_score);
	solver.def("solve_indexed_for_score", &Solver<Value, Index>::solve_indexed_for_score);
	solver.def("solve_binary_for_score", &Solver<Value, Index>::solve_binary_for_score);
	solver.def("solve_for_alignment", &Solver<Value, Index>::solve_for_alignment);
	solver.def("solve_for_alignment_iterator", &Solver<Value, Index>::solve_for_alignment_iterator);
	solver.def("solve_indexed_for_alignment", &Solver<Value, Index>::solve_indexed_for_alignment);
	solver.def("solve_indexed_for_alignment_iterator", &Solver<Value, Index>::solve_indexed_for_alignment_iterator);
	solver.def("solve_binary_for_alignment", &Solver<Value, Index>::solve_binary_for_alignment);
	solver.def("solve_binary_for_alignment_iterator", &Solver<Value, Index>::solve_binary_for_alignment_iterator);
	solver.def("solve_for_solution", &Solver<Value, Index>::solve_for_solution);
	solver.def("solve_for_solution_iterator", &Solver<Value, Index>::solve_for_solution_iterator);
	solver.def("solve_indexed_for_solution", &Solver<Value, Index>::solve_indexed_for_solution);
	solver.def("solve_indexed_for_solution_iterator", &Solver<Value, Index>::solve_indexed_for_solution_iterator);
	solver.def("solve_binary_for_solution", &Solver<Value, Index>::solve_binary_for_solution);
	solver.def("solve_binary_for_solution_iterator", &Solver<Value, Index>::solve_binary_for_solution_iterator);

	py::class_<Alignment<Index>, AlignmentRef<Index>> alignment(
		m, "Alignment", py::module_local());
	alignment.def_property_readonly("score", &Alignment<Index>::score);
	alignment.def_property_readonly("s_to_t", &Alignment<Index>::s_to_t);
	alignment.def_property_readonly("t_to_s", &Alignment<Index>::t_to_s);

	py::class_<AlignmentIterator<Index>, AlignmentIteratorRef<Index>> alignment_iterator(
		m, "AlignmentIterator", py::module_local());
	alignment_iterator.def("next", &AlignmentIterator<Index>::next);

	py::class_<Solution, SolutionRef> solution(m, "Solution", py::module_local());
	solution.def_property_readonly("values", &Solution::values);
	solution.def_property_readonly("traceback_has_max_degree_1", &Solution::traceback_has_max_degree_1);
	solution.def_property_readonly("traceback_as_matrix", &Solution::traceback_as_matrix);
	solution.def_property_readonly("traceback_as_edges", &Solution::traceback_as_edges);
	solution.def_property_readonly("path", &Solution::path);
	solution.def_property_readonly("score", &Solution::score);
	solution.def_property_readonly("alignment", &Solution::alignment);
	solution.def_property_readonly("algorithm", &Solution::algorithm);

	py::class_<SolutionIterator, SolutionIteratorRef> solution_iterator(
		m, "SolutionIterator", py::module_local());
	solution_iterator.def("next", &SolutionIterator::next);

	py::class_<Algorithm, AlgorithmRef> algorithm(
		m, "Algorithm", py::module_local());
	algorithm.def_property_readonly("name", &Algorithm::name);
	algorithm.def_property_readonly("runtime", &Algorithm::runtime);
	algorithm.def_property_readonly("memory", &Algorithm::memory);

	py::class_<Options, std::shared_ptr<Options>> options(
		m, "Options", py::module_local());
	m.def("create_options", &create_options<Options>);
}

inline void register_enum(py::module_ &m) {
	py::enum_<enums::Type>(m, "Type", py::module_local())
        .value("ALIGNMENT", enums::Type::ALIGNMENT)
        .value("DTW", enums::Type::DTW);

	py::enum_<enums::Direction>(m, "Direction", py::module_local())
        .value("MINIMIZE", enums::Direction::MINIMIZE)
        .value("MAXIMIZE", enums::Direction::MAXIMIZE);

	py::enum_<enums::Detail>(m, "Detail", py::module_local())
        .value("SCORE", enums::Detail::SCORE)
        .value("ALIGNMENT", enums::Detail::ALIGNMENT)
        .value("SOLUTION", enums::Detail::SOLUTION);

	py::enum_<enums::Count>(m, "Count", py::module_local())
        .value("ONE", enums::Count::ONE)
        .value("ALL", enums::Count::ALL);

	py::enum_<enums::Locality>(m, "Locality", py::module_local())
        .value("LOCAL", enums::Locality::LOCAL)
        .value("GLOBAL", enums::Locality::GLOBAL)
        .value("SEMIGLOBAL", enums::Locality::SEMIGLOBAL);
}

} // pyalign

#endif // __PYALIGN_PYALIGN_H__
