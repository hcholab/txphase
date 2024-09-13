#!/usr/bin/env bash

if [ "$SGX" == "1" ]; then
    ./run_gramine.sh data/datasets/ukb/giab.sh
else
    ./run.sh data/datasets/ukb/giab.sh
fi

