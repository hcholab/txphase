#!/usr/bin/env bash

DATA_DIR=data
source profiles/dataset/ukb_single.sh

TEST=tmp/phased.vcf.gz

./scripts/switch-error-rate.sh $TEST $PARENTS $TRIO tmp
