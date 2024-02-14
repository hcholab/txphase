#!/usr/bin/env bash

LITE=$1
SGX=$2
DATASET_ENV=$3
PHASING_OPTIONS=$4

source config.sh
source $DATASET_ENV

HOST_OPTIONS="\
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath tmp/phased.vcf.gz)"

killall host >/dev/null 2>&1
mkdir -p tmp

(
(cd host && cargo +nightly build $PROFILE) && \
    (cd phasing && cargo +nightly build $PROFILE $FEATURES $TARGET)) &&
    (
    (cd host && cargo +nightly run $PROFILE -- $HOST_OPTIONS) & \
        (cd phasing && cargo +nightly run $PROFILE $FEATURES $TARGET -- $PHASING_OPTIONS))
