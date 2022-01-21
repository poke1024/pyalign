#include <iostream>
#include <fstream>
#include <chrono>
#include "solver.h"

template<class T>
struct fast_vec {
    T _x[1];
    inline fast_vec() {
    }
    inline fast_vec(const T x) {
        _x[0] = x;
    }
    inline fast_vec<T> operator+(int x) const {
        return fast_vec<T>{{_x[0] + x}};
    }
    inline int size() {
        return 1;
    }
    inline T& operator()(size_t i) {
        return _x[i];
    }
    inline T operator()(size_t i) const {
        return _x[i];
    }
    inline void fill(T x) {
        _x[0] = x;
    }
};

template<typename T>
inline fast_vec<T> maximum(const fast_vec<T> &a, const fast_vec<T> &b) {
    return fast_vec<T>(std::max(a._x[0], b._x[0]));
}


struct my_cell_type {
	typedef float value_type;
	typedef int32_t index_type;

	static constexpr int batch_size = 1;

	typedef fast_vec<value_type> value_vec_type;
	typedef fast_vec<index_type> index_vec_type;
	typedef fast_vec<bool> mask_vec_type;
};

 std::string repeat(const std::string& input, size_t num) {
    std::ostringstream os;
    std::fill_n(std::ostream_iterator<std::string>(os), num, input);
    return os.str();
 }

int main() {
	pyalign::core::LinearGapCostSolver<
		//my_cell_type,
		pyalign::core::cell_type<float, int16_t>,
		pyalign::core::problem_type<
		    pyalign::core::goal::optimal_score,
		    pyalign::core::direction::maximize>,
		pyalign::core::Global> solver(
			10000, 10000,
			0, 0
		);

    std::string seq = repeat("ACGT", 1000);
    /*std::ifstream f ("genome.txt");
    assert(f.is_open());
    std::getline(f, seq);*/

    std::cout << "seq len " << seq.size() << std::endl;

	const std::string a = seq; //.substr(0, 5000);
	const std::string b = seq; //.substr(5000, 10000);

	solver.solve([&a, &b] (int i, int j) {
		if (a[i] == b[j]) {
			return 1;
		} else {
			return 0;
		}
	}, a.size(), b.size());

    using std::chrono::high_resolution_clock;

    const auto t1 = high_resolution_clock::now();
	float score = solver.score(a.size(), b.size())(0);
    const auto t2 = high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> ms_double = t2 - t1;

	std::cout << score << std::endl;
	std::cout << "time: " << ms_double.count() << std::endl;

	return 0;
}

