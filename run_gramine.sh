#!/usr/bin/env bash

DATASET_ENV=$1
PHASING_OPTIONS=$2

source config.sh
source $DATASET_ENV

mkdir -p tmp
HOST_OPTIONS="\
    --worker-port-base $PORT \
    --n-workers $N_WORKERS \
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT) \
    --output $(realpath tmp/phased.vcf.gz)"

killall host >/dev/null 2>&1

cargo +nightly build --release -p host && \
cargo +nightly build --release -p phasing $FEATURES && \
make SGX=1 EDMM=1 && \
(
for worker_id in $(seq 0 $(($N_WORKERS - 1)))
do
    gramine-sgx phasing/phasing --host-port $(($PORT + $worker_id)) $PHASING_OPTIONS &
done
target/release/host $HOST_OPTIONS
)
