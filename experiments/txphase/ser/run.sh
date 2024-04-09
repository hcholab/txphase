#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2

OUTPUT_DIR_SIZE=$(realpath $OUTPUT_DIR/$UKB_REF_PANEL_SIZE)

MAIN_DIR=$(git rev-parse --show-toplevel)

mkdir -p $OUTPUT_DIR_SIZE
cd $MAIN_DIR &&
    ./run.sh data/datasets/ukb/env_all.sh "" $OUTPUT_DIR_SIZE

./ser_ukb.sh $OUTPUT_DIR_SIZE
