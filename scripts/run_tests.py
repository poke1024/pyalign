import nose
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

nose.main(defaultTest="pyalign.tests", config=nose.config.Config(verbosity=3))
