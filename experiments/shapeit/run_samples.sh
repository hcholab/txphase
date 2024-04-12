#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2
export UKB_INPUT_SIZE=$3

MAIN_DIR=$(git rev-parse --show-toplevel)
SIZE_DIR=$OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE

source $MAIN_DIR/data/datasets/ukb/env_all.sh

N_CPUS=$(nproc --all)

mkdir -p $SIZE_DIR
/usr/bin/time -f "%e %M" -o $SIZE_DIR/time_mem.txt $MAIN_DIR/shapeit-bin/phase_common_static \
    --input $INPUT_SIZE \
    --reference $BCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --thread $N_CPUS \
    --output $SIZE_DIR/phased.bcf \
    --log $SIZE_DIR/log.txt
