#!/usr/bin/env bash

BIN_DIR=$1
DATASET=$2
PHASING_OPTIONS=$3
OUTPUT_DIR=$4

MAIN_DIR=$(git rev-parse --show-toplevel)
GRAMINE_DIR=$MAIN_DIR/experiments/sgx/gramine

source $DATASET

HOST_OPTIONS="\
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath $OUTPUT_DIR/phased.vcf.gz)"


killall host >/dev/null 2>&1

$BIN_DIR/host $HOST_OPTIONS & \
    (cd $GRAMINE_DIR && \
    /usr/bin/time -f "%e" -o $OUTPUT_DIR/time.txt gramine-sgx phasing $PHASING_OPTIONS)

