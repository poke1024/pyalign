#!/bin/bash

SCRIPTPATH=`dirname "$0"`
cd "${SCRIPTPATH}/../pyalign/tests"
python -m unittest discover
