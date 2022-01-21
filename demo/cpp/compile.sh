#!/bin/bash
cd "`dirname $0`"
mkdir -p build
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/python3.9:$CONDA_PREFIX/lib/python3.9/site-packages/numpy/core/include:`dirname $0`/../../pyalign/algorithm:`dirname $0`/../.."

#clang++ -std=c++17 -g demo1.cpp -o build/demo1
#clang++ -std=c++17 -g demo2.cpp -o build/demo2
#clang++ -std=c++17 -g demo3.cpp -o build/demo3

clang++ -std=c++17 -g -O3 perf1.cpp -o build/perf1
#clang++ -std=c++17 -g -O3 perf1.cpp -S -mllvm --x86-asm-syntax=intel -o build/perf1.s
#clang++ -std=c++17 -O3 perf1.cpp -S -o build/perf1.s
#g++ -std=c++17 -S -fverbose-asm -g -O2 perf1.cpp -o build/perf1.s
