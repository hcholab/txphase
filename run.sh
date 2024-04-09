#!/usr/bin/env bash

DATASET_ENV=$1
PHASING_OPTIONS=$2
OUTPUT_DIR=${3:-tmp/phased.vcf.gz}

source config.sh
source $DATASET_ENV

mkdir -p tmp
HOST_OPTIONS="\
    --worker-port-base $PORT \
    --n-workers $N_WORKERS \
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath $OUTPUT_DIR)"

killall host >/dev/null 2>&1

cargo +nightly build --release -p host && \
cargo +nightly build --release -p phasing $FEATURES && \
(for worker_id in $(seq 0 $(($N_WORKERS - 1)))
do
    target/release/phasing --host-port $(($PORT + $worker_id)) $PHASING_OPTIONS &
    pid[$worker_id]=$!
done
trap "kill $(printf " %d" "${pid[@]}"); exit 1" INT
target/release/host $HOST_OPTIONS
)
