#define FORCE_IMPORT_ARRAY

#include "pyalign/algorithm/pyalign.h"
#include "pyalign/algorithm/factory.h"

PYBIND11_MODULE(algorithm, m) {
	xt::import_numpy();
	pyalign::register_solver<float, int16_t>(m);
}
