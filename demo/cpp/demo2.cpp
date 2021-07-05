#include <iostream>
#include "solver.h"

int main() {
	typedef pyalign::cell_type<float, int16_t, pyalign::machine_batch_size> cell_type;

	pyalign::LinearGapCostSolver<
		cell_type,
		pyalign::problem_type<pyalign::goal::optimal_score, pyalign::direction::maximize>,
		pyalign::Global> solver(
			0, 0,
			100, 100
		);

	std::cout << "batch size: " << solver.batch_size() << std::endl;

	const std::string a = "INDUSTRY";
	const std::string b = "INTEREST";

	solver.solve([&a, &b] (int i, int j) {
		cell_type::value_vec_type v;
		if (a[i] == b[j]) {
			v.fill(1);
		} else {
			v.fill(0);
		}
		return v;
	}, a.size(), b.size());

	const auto r = solver.score(a.size(), b.size());
	for (int i = 0; i < r.shape(0); i++) {
		std::cout << r(i) << std::endl;
	}
}
