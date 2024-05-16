#!/usr/bin/env bash

if [ "$SGX" == "1" ]; then
    ./run_gramine.sh data/datasets/hrc/env.sh "$PHASING_OPTIONS"
else
    ./run.sh data/datasets/hrc/env.sh "$PHASING_OPTIONS"
fi

