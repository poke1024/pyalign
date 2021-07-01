#include <iostream>
#include "solver.h"

int main() {
	pyalign::LinearGapCostSolver<
		pyalign::cell_type<float, int16_t>,
		pyalign::problem_type<pyalign::goal::optimal_score, pyalign::direction::maximize>,
		pyalign::Global> solver(
			0, 0,
			100, 100
		);

	const std::string a = "INDUSTRY";
	const std::string b = "INTEREST";

	solver.solve([&a, &b] (int i, int j) {
		if (a[i] == b[i]) {
			return 1;
		} else {
			return 0;
		}
	}, a.size(), b.size());

	std::cout << solver.score(a.size(), b.size())(0) << std::endl;

	return 0;
}

