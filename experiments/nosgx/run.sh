#!/usr/bin/env bash

BIN_DIR=$1
DATASET=$2
PHASING_OPTIONS=$3
OUTPUT_DIR=$4

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_DIR=$(git rev-parse --show-toplevel)

DATA_DIR=$MAIN_DIR/data

source $DATASET

HOST_OPTIONS="\
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath $OUTPUT_DIR/phased.vcf.gz)"

$BIN_DIR/host $HOST_OPTIONS & \
    /usr/bin/time -f "%e %M" -o $OUTPUT_DIR/time_mem.txt $BIN_DIR/phasing $PHASING_OPTIONS

