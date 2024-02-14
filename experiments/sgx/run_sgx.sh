#!/usr/bin/env bash

BIN_DIR=$1
DATASET=$2
PHASING_OPTIONS=$3
OUTPUT_DIR=$4
CARGO_MANIFEST_DIR=$5

source $DATASET

HOST_OPTIONS="\
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath $OUTPUT_DIR/phased.vcf.gz)"

export CARGO_MANIFEST_DIR=$(realpath $CARGO_MANIFEST_DIR)

killall host >/dev/null 2>&1

$BIN_DIR/host $HOST_OPTIONS & \
    /usr/bin/time -f "%e" -o $OUTPUT_DIR/time.txt ftxsgx-runner-cargo $BIN_DIR/phasing $PHASING_OPTIONS

