#!/usr/bin/env bash

PHASING_OPTIONS="--max-m3vcf-unique-haps=300 --min-m3vcf-unique-haps=200"

if [ "$SGX" == "1" ]; then
    ./run_gramine.sh data/datasets/1kg/env.sh "$PHASING_OPTIONS"
else
    ./run.sh data/datasets/1kg/env.sh "$PHASING_OPTIONS"
fi

