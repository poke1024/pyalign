set -xe
python -VV
python -m site
python -m pip install --upgrade pip "setuptools>=42" "wheel" "pybind11~=2.6" "numpy~=1.19" "pyyaml>=5.4.1"
