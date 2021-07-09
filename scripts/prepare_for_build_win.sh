choco install python --version 3.8.6
export PATH="/c/Python38:/c/Python38/Scripts:$PATH"
ln -s /c/Python38/python.exe /c/Python38/python3.exe
choco install llvm
cc --version
choco install git cmake
pip install "pybind11[global]~=2.6.2" "numpy~=1.19"
python scripts/install_xtensor.py xtl 0.7.2
python scripts/install_xtensor.py xtensor 0.23.10
python scripts/install_xtensor.py xsimd 7.4.9
python scripts/install_xtensor.py xtensor-python 0.25.3
