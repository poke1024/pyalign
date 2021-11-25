#ifndef __PYALIGN_OPTIONS_H__
#define __PYALIGN_OPTIONS_H__ 1

#include "pyalign/algorithm/common.h"
#include "pyalign/algorithm/enum.h"

namespace pyalign {

template<typename Value, typename Index>
class Options {
public:
	typedef Value value_type;
	typedef Index index_type;

private:
	struct base {
		inline base(const py::dict &p_options) :
			type(p_options["solver"].cast<enums::Type>()),
			batch(p_options["batch"].cast<bool>()),
			direction(p_options["direction"].cast<enums::Direction>()),
			remove_dup(!p_options["return_dup"].cast<bool>()) {
		}

		const enums::Type type;
		const bool batch;
		const enums::Direction direction;
		const bool remove_dup;
	};

	struct alignment {
		inline alignment(const py::dict &p_options) :
			detail(p_options["codomain"].attr("detail").cast<enums::Detail>()),
			count(p_options["codomain"].attr("count").cast<enums::Count>()),
			locality(p_options.contains("locality") ?
				p_options["locality"].cast<enums::Locality>() : enums::Locality::LOCAL),
			gap_costs(p_options.contains("gap_cost") ?
				p_options["gap_cost"] : py::none().cast<py::object>()) {
		}

		const enums::Detail detail;
		const enums::Count count;
		const enums::Locality locality;
		const GapCosts<Value> gap_costs;
	};

	const py::dict m_options;
	const base m_base;
	const std::optional<alignment> m_alignment;

public:
	inline Options(
		const py::dict &p_options) :

		m_options(p_options),
		m_base(p_options),
		m_alignment(
			p_options["solver"].cast<enums::Type>() == enums::Type::ALIGNMENT ?
				alignment(p_options) : std::optional<alignment>()
		) {
	}

	inline py::dict to_dict() {
		return m_options;
	}

	inline enums::Type type() const {
		return m_base.type;
	}

	inline bool batch() const {
		return m_base.batch;
	}

	inline enums::Direction direction() const {
		return m_base.direction;
	}

	inline bool remove_dup() const {
	    return m_base.remove_dup;
	}

	inline enums::Detail detail() const {
		if (m_alignment.has_value()) {
			return m_alignment->detail;
		} else {
			throw std::runtime_error("detail not available");
		}
	}

	inline enums::Count count() const {
		if (m_alignment.has_value()) {
			return m_alignment->count;
		} else {
			throw std::runtime_error("count not available");
		}
	}

	inline enums::Locality locality() const {
		if (m_alignment.has_value()) {
			return m_alignment->locality;
		} else {
			throw std::runtime_error("locality not available");
		}
	}

	inline const GapCosts<Value> &gap_costs() const {
		if (m_alignment.has_value()) {
			return m_alignment->gap_costs;
		} else {
			throw std::runtime_error("gap_costs not available");
		}
	}

	inline std::shared_ptr<Options<Value, Index>> clone() const {
		return std::make_shared<Options<Value, Index>>(m_options);
	}
};

template<typename Value, typename Index>
using OptionsRef = std::shared_ptr<Options<Value, Index>>;

} // pyalign

#endif // __PYALIGN_OPTIONS_H__
