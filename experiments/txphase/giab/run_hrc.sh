#!/usr/bin/env bash

OUTPUT_DIR=$1

MAIN_DIR=$(git rev-parse --show-toplevel)
source $MAIN_DIR/config.sh
source $MAIN_DIR/data/datasets/hrc/env.sh

(
cd $MAIN_DIR &&
cargo +nightly build --release -p host && \
cargo +nightly build --release -p phasing $FEATURES
)

N_REPEATS=100
for ((n=0;n<$N_REPEATS;n++)); do
    OUTPUT=$OUTPUT_DIR/phased_$n.vcf.gz
    port=$(expr $PORT + $n)
    PHASING_OPTIONS="--min-het-rate=0.6 \
        --prg-seed $RANDOM"
    HOST_OPTIONS="\
        --worker-port-base $port \
        --n-workers 1 \
        --ref-panel $(realpath $M3VCF_REF_PANEL) \
        --genetic-map $(realpath $GMAP) \
        --input $(realpath $INPUT_SIZE) \
        --output $(realpath $OUTPUT)"
    echo "$MAIN_DIR/target/release/phasing --host-port $port $PHASING_OPTIONS & \
    $MAIN_DIR/target/release/host $HOST_OPTIONS"
done | xargs -P $(nproc --all) -I {} bash -c {}



