#ifndef __PYALIGN_H__
#define __PYALIGN_H__ 1

#define XTENSOR_USE_XSIMD 1

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xsort.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <xtensor-python/pytensor.hpp>

#ifndef PYALIGN_FEATURES_DTW
#define PYALIGN_FEATURES_DTW 1
#endif

#ifndef PYALIGN_FEATURES_SCORE_ONLY
#define PYALIGN_FEATURES_SCORE_ONLY 1
#endif

#ifndef PYALIGN_FEATURES_MINIMIZE
#define PYALIGN_FEATURES_MINIMIZE 1
#endif

#ifndef PYALIGN_FEATURES_ALL_SOLUTIONS
#define PYALIGN_FEATURES_ALL_SOLUTIONS 1
#endif

#endif // __PYALIGN_H__
