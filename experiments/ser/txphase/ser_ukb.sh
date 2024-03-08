#!/usr/bin/env bash

OUTPUT_DIR=$1

MAIN_DIR=$(git rev-parse --show-toplevel)

source $MAIN_DIR/data/datasets/ukb/env_all.sh

TEST=$OUTPUT_DIR/phased.vcf.gz

$MAIN_DIR/scripts/switch-error-rate.sh $TEST $PARENTS $TRIO $OUTPUT_DIR
