#define FORCE_IMPORT_ARRAY

#include "pyalign/algorithm/pyalign.h"
#include "pyalign/algorithm/factory.h"
#include "pyalign/algorithm/options.h"

PYBIND11_MODULE(algorithm, m) {
	xt::import_numpy();

	auto m16 = m.def_submodule("m16");
	pyalign::register_solver<pyalign::Options<float, int16_t>>(m16);

    auto m32 = m.def_submodule("m32");
	pyalign::register_solver<pyalign::Options<float, int32_t>>(m32);

    pyalign::register_algorithm(m);
	pyalign::register_enum(m);
}
