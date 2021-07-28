#define FORCE_IMPORT_ARRAY

#include "pyalign/algorithm/pyalign.h"
#include "pyalign/algorithm/factory.h"
#include "pyalign/algorithm/options.h"

PYBIND11_MODULE(algorithm, m) {
	xt::import_numpy();
	pyalign::register_solver<pyalign::Options<float, int16_t>>(m);
	pyalign::register_enum(m);
}
