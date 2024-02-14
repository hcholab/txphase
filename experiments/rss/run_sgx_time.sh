#!/usr/bin/env bash

OUTPUT_DIR=$1
USE_RSS=$2
PBWT_DEPTH=$3

MAIN_DIR=$(git rev-parse --show-toplevel)

HEAP_SIZE=0xC0000000

if [[ $USE_RSS -eq 1 ]]
then
    PHASING_OPTIONS="--use-rss --max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100 --pbwt-depth $PBWT_DEPTH"
else
    PHASING_OPTIONS="--max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100 --pbwt-depth $PBWT_DEPTH"
fi

echo "[package.metadata.fortanix-sgx]
stack-size=0x200000
heap-size=$HEAP_SIZE
threads=1" > $OUTPUT_DIR/Cargo.toml

$MAIN_DIR/experiments/sgx/build_sgx.sh bin_sgx && \
    $MAIN_DIR/experiments/sgx/run_sgx.sh bin_sgx $MAIN_DIR/data/datasets/1kg/env.sh "$PHASING_OPTIONS" $OUTPUT_DIR $OUTPUT_DIR | tee $OUTPUT_DIR/log.txt
