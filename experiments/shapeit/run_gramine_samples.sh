#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2
export UKB_INPUT_SIZE=$3

MAIN_DIR=$(git rev-parse --show-toplevel)
source $MAIN_DIR/data/datasets/ukb/env_all.sh

SIZE_DIR=$OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE
mkdir -p $SIZE_DIR
SIZE_DIR=$(realpath $OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE)

N_CPUS=$(nproc --all)

cd $MAIN_DIR/shapeit-bin
/usr/bin/time -f "%e" -o $SIZE_DIR/time.txt gramine-sgx phase_common_static \
    --input $INPUT_SIZE \
    --reference $BCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --thread $N_CPUS \
    --output $SIZE_DIR/phased.bcf \
    --log $SIZE_DIR/log.txt
