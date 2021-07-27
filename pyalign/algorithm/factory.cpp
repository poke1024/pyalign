#include "pyalign/algorithm/factory.h"

namespace pyalign {

OptionsRef create_options(const py::dict &p_options) {
	if (p_options["solver"].cast<Options::Type>() == Options::Type::ALIGNMENT) {
		return std::make_shared<AlignmentOptions>(p_options);
	} else {
		return std::make_shared<Options>(p_options);
	}
}

} // pyalign
