import nose
import os
import sys
import traceback

os.chdir(os.path.dirname(os.path.realpath(__file__)))

try:
	success = nose.run(
		defaultTest="pyalign.tests",
		config=nose.config.Config(verbosity=3))
except:
	traceback.print_exc()
	raise

if success:
	print("tests ok.")
	sys.exit(0)
else:
	print("tests failed.")
	sys.exit(1)
