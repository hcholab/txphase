#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2

OUTPUT_DIR_SIZE=$OUTPUT_DIR/$UKB_REF_PANEL_SIZE

MAIN_DIR=$(git rev-parse --show-toplevel)

N_CPUS=$(cat /proc/cpuinfo | grep processor | wc -l)

PHASING_OPTIONS="--n-cpus $N_CPUS --single-sample=false"

killall host >/dev/null 2>&1
mkdir -p bin
mkdir -p $OUTPUT_DIR_SIZE
$MAIN_DIR/experiments/nosgx/build.sh bin && \
    $MAIN_DIR/experiments/nosgx/run.sh bin $MAIN_DIR/data/datasets/ukb/env_all.sh "$PHASING_OPTIONS" $OUTPUT_DIR_SIZE $OUTPUT_DIR_SIZE | tee $OUTPUT_DIR_SIZE/log.txt

./ser_ukb.sh $OUTPUT_DIR_SIZE
