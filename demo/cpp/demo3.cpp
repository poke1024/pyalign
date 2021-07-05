#include <iostream>
#include "solver.h"

class Alignment {
public:
	std::vector<std::pair<size_t, size_t>> edges;
	float score;

	inline void resize(const size_t len_s, const size_t len_t) {
	}

	inline void add_edge(const size_t u, const size_t v) {
		edges.push_back(std::make_pair(u, v));
	}

	inline void set_score(const float p_score) {
		score = p_score;
	}
};

int main() {
	typedef pyalign::cell_type<float, int16_t> cell_type;

	pyalign::AffineGapCostSolver<
		cell_type,
		pyalign::problem_type<pyalign::goal::all_optimal_alignments, pyalign::direction::minimize>,
		pyalign::Global> solver(
			pyalign::AffineCost<float>(1, 4), pyalign::AffineCost<float>(1, 4),
			20, 20
		);

	std::cout << "batch size: " << solver.batch_size() << std::endl;

	const std::string a = "CC";
	const std::string b = "ACCT";

	solver.solve([&a, &b] (int i, int j) {
		cell_type::value_vec_type v;
		if (a[i] == b[j]) {
			v.fill(0);
		} else {
			v.fill(1);
		}
		return v;
	}, a.size(), b.size());

	const auto r = solver.alignment_iterator<Alignment>(a.size(), b.size());
	for (int i = 0; i < r.size(); i++) {
		auto it = r[i];
		int j = 0;
		while (true) {
			auto alignment = it->next();
			if (!alignment.get()) {
				break;
			}
			std::cout << "[" << j << "] " << alignment->score << std::endl;
			for (const auto &e : alignment->edges) {
				std::cout << "(" << std::get<0>(e) << "," << std::get<1>(e) << ")" << std::endl;
			}
			std::cout << std::endl;
			j++;
		}
	}

	return 0;
}
