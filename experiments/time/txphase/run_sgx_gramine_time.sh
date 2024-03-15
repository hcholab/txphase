#!/usr/bin/env bash

OUTPUT_DIR=$1
USE_RSS=$2
export UKB_REF_PANEL_SIZE=$3
export UKB_SAMPLE_ID=4197007

OUTPUT_DIR=$(realpath $OUTPUT_DIR)
OUTPUT_DIR_SIZE=$OUTPUT_DIR/$UKB_REF_PANEL_SIZE

MAIN_DIR=$(git rev-parse --show-toplevel)

DATASET=$MAIN_DIR/data/datasets/ukb/env_single.sh

if [[ $USE_RSS -eq 1 ]]
then
    PHASING_OPTIONS="--use-rss"
fi

killall host >/dev/null 2>&1
mkdir -p $OUTPUT_DIR_SIZE
mkdir -p bin
$MAIN_DIR/experiments/sgx/build_sgx_gramine.sh bin && \
    $MAIN_DIR/experiments/sgx/run_sgx_gramine.sh bin $DATASET "$PHASING_OPTIONS" $OUTPUT_DIR_SIZE $OUTPUT_DIR_SIZE | tee $OUTPUT_DIR_SIZE/log.txt
