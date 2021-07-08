yum install -y devtoolset-9-toolchain
scl enable devtoolset-9 bash
cc --version
yum install -y git cmake
cd $(dirname $0)
./prepare_for_build.sh
