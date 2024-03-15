#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2
export UKB_SAMPLE_ID=4197007

OUTPUT_DIR=$(realpath $OUTPUT_DIR)
OUTPUT_DIR_SIZE=$OUTPUT_DIR/$UKB_REF_PANEL_SIZE

MAIN_DIR=$(git rev-parse --show-toplevel)

DATASET=$MAIN_DIR/data/datasets/ukb/env_single.sh

source $DATASET

mkdir -p $OUTPUT_DIR_SIZE

/usr/bin/time -f "%e %M" -o $OUTPUT_DIR_SIZE/time.txt $MAIN_DIR/shapeit-bin/phase_common_static \
    --input $INPUT \
    --reference $VCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --filter-maf 0.001 \
    --thread 1 \
    --output $OUTPUT_DIR_SIZE/phased.bcf \
    --log $OUTPUT_DIR_SIZE/log.txt

