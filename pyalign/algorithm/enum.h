#ifndef __PYALIGN_ENUM_H__
#define __PYALIGN_ENUM_H__ 1

namespace pyalign {

namespace enums {

	enum struct Type {
		ALIGNMENT,
		DTW
	};

	enum struct Direction {
		MINIMIZE,
		MAXIMIZE
	};

	enum struct Detail {
		SCORE,
		ALIGNMENT,
		SOLUTION
	};

	enum struct Count {
		ONE,
		ALL
	};

	enum struct Locality {
		LOCAL,
		GLOBAL,
		SEMIGLOBAL
	};

} // enum

} // pyalign

#endif // __PYALIGN_ENUM_H__
