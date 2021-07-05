#!/bin/bash
cd "`dirname $0`"
mkdir -p build
export CPLUS_INCLUDE_PATH="/opt/miniconda3/envs/pyalign/include:`dirname $0`/../../pyalign/algorithm"
clang++ -std=c++17 -g demo1.cpp -o build/demo1
clang++ -std=c++17 -g demo2.cpp -o build/demo2
clang++ -std=c++17 -g demo3.cpp -o build/demo3
