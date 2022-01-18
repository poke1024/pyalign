import importlib
import importlib.util
import os
import logging
import inspect
import pyalign


def has_avx2():
	import cpufeature
	return cpufeature.CPUFeature["AVX2"]


def has_apple_m1():
	import cpuinfo
	import re
	brand = cpuinfo.get_cpu_info().get('brand_raw')
	return re.match("^Apple M1", brand) is not None


def import_algorithm():
	if os.environ.get('PYALIGN_PDOC') is not None:
		return None

	candidates = (
		('native', lambda: True),
		('intel_avx2', has_avx2),
		('apple_m1', has_apple_m1),
		('generic', lambda: True)
	)

	missing_module_names = []

	for name, check in candidates:
		module_name = f"pyalign.algorithm.{name}"
		if importlib.util.find_spec(module_name) is not None:
			if check():
				logging.info(f"running in {name} mode.")
				return importlib.import_module(module_name + ".algorithm")
			else:
				logging.info(f"{name} found, but not suitable.")
		else:
			missing_module_names.append(module_name)

	logging.info(f"pyalign installation is at {inspect.getfile(pyalign)}")

	raise RuntimeError(f"none of {missing_module_names} is available")
