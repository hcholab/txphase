#!/usr/bin/env bash

export UKB_REF_PANEL_SIZE=$1
export UKB_INPUT_SIZE=$2

if [ "$SGX" == "1" ]; then
    ./run_gramine.sh data/datasets/ukb/env_all.sh "$PHASING_OPTIONS"
else
    ./run.sh data/datasets/ukb/env_all.sh "$PHASING_OPTIONS"
fi

