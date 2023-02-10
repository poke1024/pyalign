# specifying a wrong numpy version here will end up in runtime errors like:
# module compiled against API version 0xe but this version of numpy is 0xd
python -m ensurepip
python -m pip install -U pip wheel
python -m pip install "pybind11[global]~=2.6.2" "numpy~=1.19" cmake
python ci/install_xtensor.py xtl 0.7.2
python ci/install_xtensor.py xsimd 7.4.9
python ci/install_xtensor.py xtensor 0.23.10
python ci/install_xtensor.py xtensor-python 0.25.3
