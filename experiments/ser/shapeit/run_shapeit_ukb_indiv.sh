#!/usr/bin/env bash

export UKB_SAMPLE_ID=$1
export UKB_REF_PANEL_SIZE=$2
TAR_DIR=$3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=$MAIN_DIR/data
SAMPLE_DIR=$TAR_DIR/$UKB_SAMPLE_ID

source $MAIN_DIR/profiles/dataset/ukb_single.sh

mkdir -p $SAMPLE_DIR && \
/usr/bin/time -f "%e %M" -o $SAMPLE_DIR/time.txt $MAIN_DIR/shapeit-bin/phase_common_static \
    --input $INPUT \
    --reference $VCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --filter-maf 0.001 \
    --thread 1 \
    --output $SAMPLE_DIR/phased.bcf \
    --log $SAMPLE_DIR/log.txt && \

$MAIN_DIR/scripts/switch-error-rate.sh $SAMPLE_DIR/phased.bcf $PARENTS $TRIO $SAMPLE_DIR
