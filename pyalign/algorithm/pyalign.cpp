#define FORCE_IMPORT_ARRAY
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "solver.h"

namespace py = pybind11;


class Alignment {
public:
	typedef int16_t Index;

private:
	xt::pyarray<Index> m_s_to_t;
	xt::pyarray<Index> m_t_to_s;
	float m_score;

public:
	inline void resize(const size_t len_s, const size_t len_t) {
		m_s_to_t.resize({len_s});
		m_s_to_t.fill(-1);

		m_t_to_s.resize({len_t});
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

	inline const xt::pyarray<Index> &s_to_t() const {
		return m_s_to_t;
	}

	inline const xt::pyarray<Index> &t_to_s() const {
		return m_t_to_s;
	}
};

typedef std::shared_ptr<Alignment> AlignmentRef;


class Solver {
public:
	virtual inline ~Solver() {
	}

	virtual AlignmentRef solve(
		const xt::pyarray<float> &similarity) const = 0;
};

typedef std::shared_ptr<Solver> SolverRef;


template<typename S>
class SolverImpl : public Solver {
private:
	S m_solver;

public:
	template<typename... Args>
	inline SolverImpl(const Args&... args) : m_solver(args...) {
	}

	virtual AlignmentRef solve(
		const xt::pyarray<float> &similarity) const override {

		const auto alignment = std::make_shared<Alignment>();

		const float score = m_solver.solve(
			*alignment.get(),
			similarity,
			similarity.shape(0),
			similarity.shape(1));

		alignment->set_score(score);
		return alignment;
	}
};

inline xt::pyarray<float> default_gap_tensor(const size_t p_len) {
	xt::pyarray<float> w;
	w.resize({p_len});
	w.fill(0);
	return w;
}

inline alignments::GapTensorFactory to_gap_tensor_factory(const py::object &p_gap) {
	if (p_gap.is_none()) {
		return default_gap_tensor;
	} else {
		return p_gap.attr("costs").cast<alignments::GapTensorFactory>();
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
		const auto locality = alignments::Local<float>(0);

		return std::make_shared<SolverImpl<alignments::GeneralGapCostSolver<
			alignments::Local<float>, Alignment::Index>>>(
				locality,
				to_gap_tensor_factory(gap_s),
				to_gap_tensor_factory(gap_t),
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

	py::class_<Alignment, AlignmentRef> alignment(m, "Alignment");
	alignment.def_property_readonly("score", &Alignment::score);
	alignment.def_property_readonly("s_to_t", &Alignment::s_to_t);
	alignment.def_property_readonly("t_to_s", &Alignment::t_to_s);
}
