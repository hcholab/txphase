#!/usr/bin/env bash

PHASING_OPTIONS="--max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100"
./run.sh 0 0 data/datasets/1kg/env.sh "$PHASING_OPTIONS"
