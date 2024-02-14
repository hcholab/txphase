#!/usr/bin/env bash

BIN_DIR=$1
DATASET=$2
PHASING_OPTIONS=$3
CARGO_MANIFEST_DIR=$4
OUTPUT_DIR=$5

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_DIR=$(git rev-parse --show-toplevel)

DATA_DIR=$MAIN_DIR/data

source $DATASET

HOST_OPTIONS="\
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath tmp/phased.vcf.gz)"

export CARGO_MANIFEST_DIR=$(realpath $CARGO_MANIFEST_DIR)

$BIN_DIR/host $HOST_OPTIONS & \
    ftxsgx-runner-cargo $BIN_DIR/phasing $PHASING_OPTIONS

