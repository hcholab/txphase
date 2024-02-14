#!/usr/bin/env bash

LITE=$1
SGX=$2
DATA_DIR=$3
DATASET=$4
PHASING_PROFILE=$5

source config.sh
source $DATASET
source $PHASING_PROFILE

HOST_OPTIONS="\
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath tmp/phased.vcf.gz)"

(
(cd host && cargo +nightly build $PROFILE) && \
    (cd phasing && cargo +nightly build $PROFILE $FEATURES $TARGET)) &&
    (
    (cd host && cargo +nightly run $PROFILE -- $HOST_OPTIONS) & \
        (cd phasing && cargo +nightly run $PROFILE $FEATURES $TARGET -- $PHASING_OPTIONS))
