#include <iostream>
#include <fstream>
#include <chrono>
#include "solver.h"

 std::string repeat(const std::string& input, size_t num) {
    std::ostringstream os;
    std::fill_n(std::ostream_iterator<std::string>(os), num, input);
    return os.str();
 }

template<typename F>
double benchmark(const F &f) {
    using std::chrono::high_resolution_clock;
    const auto t1 = high_resolution_clock::now();
    f();
    const auto t2 = high_resolution_clock::now();

    auto ns1 = t1.time_since_epoch().count();
    auto ns2 = t2.time_since_epoch().count();

    const auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    return s.count();
}

/*template<typename F>
double benchmark(const F &f) {
    using std::chrono::high_resolution_clock;
    const auto t1 = high_resolution_clock::now();
    f();
    const auto t2 = high_resolution_clock::now();
    const auto s = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    return s.count() / 1000.0f;
}*/

/*template <class T, class FSH>
using tiny_vector = xt::xtensor_fixed<T, FSH, xt::layout_type::row_major, false>;*/

void test_xt() {
    typedef pyalign::core::tiny_vector<float, xt::xshape<3>> cell_type;
    xt::xtensor<cell_type, 3> data;
    for (int i = 0; i < 10; i++) {
        std::cout << benchmark([&data] () {
            data.resize({2, 20000, 20000});
        }) << std::endl;
        std::cout << benchmark([&data] () {
            data.resize({0, 0, 0});
        }) << std::endl;
    }

    xt::xtensor<float, 4> data2;
    std::cout << benchmark([&data2] () {
        data2.resize({2, 20000, 20000, 3});
    }) << std::endl;

    std::cout << benchmark([&data2] () {
        data2.fill(0);
    }) << std::endl;
}

int main() {
    /*test_xt();
    return 0;*/

    std::cout << "setting up solver." << std::endl;

	pyalign::core::LinearGapCostSolver<
		//my_cell_type,
	    pyalign::core::cell_type<float, int32_t>,
		pyalign::core::problem_type<
		    pyalign::core::goal::optimal_score,
		    pyalign::core::direction::maximize>,
		pyalign::core::Global> solver(
			20000, 20000,
			0, 0
		);

    std::cout << "done setting up solver." << std::endl;

    std::string seq; // = repeat("ACGT", 1000);

    std::ifstream f ("genome.txt");
    assert(f.is_open());
    std::getline(f, seq);

    std::cout << "seq len " << seq.size() << std::endl;

	const std::string a = seq.substr(0, 5000);
	const std::string b = seq.substr(5000, 15000);

	solver.solve([&a, &b] (int i, int j) {
		if (a[i] == b[j]) {
			return 1;
		} else {
			return 0;
		}
	}, a.size(), b.size());

    std::cout << "running solve." << std::endl;

    float score;
	const auto time = benchmark([&solver, &a, &b, &score] () {
	    score = solver.score(a.size(), b.size())(0);
	});

	std::cout << score << std::endl;
	std::cout << "time: " << time << std::endl;

    std::cout << "done." << std::endl;

	return 0;
}

