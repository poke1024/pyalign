#define FORCE_IMPORT_ARRAY

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <xtensor-python/pytensor.hpp>

#include "solver.h"

namespace py = pybind11;


class Alignment {
public:
	typedef int16_t Index;

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

class Solution {
public:
	typedef float Value;
	typedef Alignment::Index Index;

private:
	const alignments::SolutionRef<Index, Value> m_solution;

public:
	Solution(const alignments::SolutionRef<Index, Value> p_solution) : m_solution(p_solution) {
	}

	xt::pytensor<Value, 2> values() const {
		return m_solution->values();
	}

	xt::pytensor<Index, 3> traceback() const {
		return m_solution->traceback();
	}

	xt::pytensor<Index, 2> path() const {
		return m_solution->path();
	}

	auto score() const {
		return m_solution->score();
	}
};

typedef std::shared_ptr<Solution> SolutionRef;

class Solver {
public:
	virtual inline ~Solver() {
	}

	virtual void solve(
		const xt::pytensor<float, 2> &similarity) const = 0;

	virtual float score() const = 0;

	virtual AlignmentRef alignment() const = 0;

	virtual SolutionRef solution() const = 0;
};

typedef std::shared_ptr<Solver> SolverRef;


template<typename S>
class SolverImpl : public Solver {
private:
	S m_solver;
	mutable size_t m_len_s; // of last problem solved
	mutable size_t m_len_t; // of last problem solved

public:
	template<typename... Args>
	inline SolverImpl(const Args&... args) :
		m_solver(args...), m_len_s(0), m_len_t(0) {
	}

	virtual void solve(
		const xt::pytensor<float, 2> &similarity) const override {

		m_len_s = similarity.shape(0);
		m_len_t = similarity.shape(1);

		m_solver.solve(
			similarity,
			m_len_s,
			m_len_t);
	}

	virtual float score() const override {
		return m_solver.score(m_len_s, m_len_t);
	}

	virtual AlignmentRef alignment() const override {
		const auto alignment = std::make_shared<Alignment>();
		const float score = m_solver.alignment(
			m_len_s, m_len_t, *alignment.get());
		alignment->set_score(score);
		return alignment;
	}

	virtual SolutionRef solution() const override {
		return std::make_shared<Solution>(
			m_solver.solution(m_len_s, m_len_t));
	}
};

inline xt::pytensor<float, 1> zero_gap_tensor(const size_t p_len) {
	xt::pytensor<float, 1> w;
	w.resize({static_cast<ssize_t>(p_len)});
	w.fill(0);
	return w;
}

inline alignments::GapTensorFactory<float> to_gap_tensor_factory(const py::object &p_gap) {
	if (p_gap.is_none()) {
		return zero_gap_tensor;
	} else {
		return p_gap.attr("costs").cast<alignments::GapTensorFactory<float>>();
	}
}

struct GapCostSpecialCases {
	inline GapCostSpecialCases(const py::object &p_gap) {
		if (p_gap.is_none()) {
			affine = 0.0f;
		} else {
	        const py::dict cost = p_gap.attr("to_special_case")().cast<py::dict>();
	        if (!cost.contains("affine")) {
	            return;
	        } else {
	            affine = cost["affine"].cast<float>();
	        }
	    }
	}

	std::optional<float> affine;
};

template<typename Locality>
SolverRef create_solver_instance(
	const Locality &p_locality,
	const py::object &p_gap_s,
	const py::object &p_gap_t,
	const size_t p_max_len_s,
	const size_t p_max_len_t) {

	const GapCostSpecialCases x_gap_s(p_gap_s);
	const GapCostSpecialCases x_gap_t(p_gap_t);

	if (x_gap_s.affine.has_value() && x_gap_t.affine.has_value()) {

		return std::make_shared<SolverImpl<alignments::AffineGapCostSolver<
			Locality, Alignment::Index>>>(
				p_locality,
				x_gap_s.affine.value(),
				x_gap_t.affine.value(),
				p_max_len_s,
				p_max_len_t);

	} else {

		return std::make_shared<SolverImpl<alignments::GeneralGapCostSolver<
			Locality, Alignment::Index>>>(
				p_locality,
				to_gap_tensor_factory(p_gap_s),
				to_gap_tensor_factory(p_gap_t),
				p_max_len_s,
				p_max_len_t);
	}
}

SolverRef create_solver(
	const size_t p_max_len_s,
	const size_t p_max_len_t,
	const py::dict &p_options) {

	const std::string locality = p_options.contains("locality") ?
		p_options["locality"].cast<std::string>() : "local";

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

	if (locality == "local") {
		const float zero = p_options.contains("zero") ?
			p_options["zero"].cast<float>() : 0.0f;

		const auto locality = alignments::Local<float>(zero);

		return create_solver_instance(
			locality,
			gap_s,
			gap_t,
			p_max_len_s,
			p_max_len_t);
	} else {
		throw std::invalid_argument(locality);
	}
}

PYBIND11_MODULE(algorithm, m) {
	xt::import_numpy();

	m.def("create_solver", &create_solver);

	py::class_<Solver, SolverRef> solver(m, "Solver");
	solver.def("solve", &Solver::solve);
	solver.def_property_readonly("score", &Solver::score);
	solver.def_property_readonly("alignment", &Solver::alignment);
	solver.def_property_readonly("solution", &Solver::solution);

	py::class_<Alignment, AlignmentRef> alignment(m, "Alignment");
	alignment.def_property_readonly("score", &Alignment::score);
	alignment.def_property_readonly("s_to_t", &Alignment::s_to_t);
	alignment.def_property_readonly("t_to_s", &Alignment::t_to_s);

	py::class_<Solution, SolutionRef> solution(m, "Solution");
	solution.def_property_readonly("values", &Solution::values);
	solution.def_property_readonly("traceback", &Solution::traceback);
	solution.def_property_readonly("path", &Solution::path);
	solution.def_property_readonly("score", &Solution::score);
}
