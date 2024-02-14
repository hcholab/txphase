#!/usr/bin/env bash

source data/datasets/ukb/env_all.sh

TEST=tmp/phased.vcf.gz

./scripts/switch-error-rate.sh $TEST $PARENTS $TRIO tmp
