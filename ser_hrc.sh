#!/usr/bin/env bash

source data/datasets/hrc/env.sh

TEST=tmp/phased.vcf.gz

./scripts/switch-error-rate.sh $TEST $PARENTS $TRIO tmp
