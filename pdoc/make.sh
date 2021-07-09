#!/bin/bash
cd "$(dirname "$0")"
export PYALIGN_PDOC=1
export PYTHONPATH=$PYTHONPATH:"$(dirname "$0")/../"
pdoc --html --output-dir build -c latex_math=True --force "../pyalign"
