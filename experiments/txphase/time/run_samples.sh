#!/usr/bin/env bash

OUTPUT_DIR=$1
UKB_REF_PANEL_SIZE=$2
UKB_INPUT_SIZE=$3

OUTPUT_DIR_SIZE=$OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE

MAIN_DIR=$(git rev-parse --show-toplevel)

source $MAIN_DIR/config.sh

mkdir -p $OUTPUT_DIR_SIZE
/usr/bin/time -f "%e" -o $OUTPUT_DIR_SIZE/time.txt ./run_internal.sh $OUTPUT_DIR $UKB_REF_PANEL_SIZE $UKB_INPUT_SIZE
