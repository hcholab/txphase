#!/usr/bin/env bash

TAR_DIR=$1
USE_RSS=$2
PBWT_DEPTH=$3
HEAP_SIZE=0xC0000000

MAIN_DIR=$(git rev-parse --show-toplevel)

if [[ $USE_RSS -eq 1 ]]
then
    PHASING_OPTIONS="--use-rss --max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100 --pbwt-depth $PBWT_DEPTH"
else
    PHASING_OPTIONS="--max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100 --pbwt-depth $PBWT_DEPTH"
fi

echo "[package.metadata.fortanix-sgx]
stack-size=0x200000
heap-size=$HEAP_SIZE
threads=9" > $TAR_DIR/Cargo.toml

$MAIN_DIR/experiments/sgx/build_sgx.sh bin
/usr/bin/time -f "%e" -o $TAR_DIR/time.txt $MAIN_DIR/experiments/sgx/run_sgx.sh bin $MAIN_DIR/profiles/dataset/1kg.sh "$PHASING_OPTIONS" $TAR_DIR $TAR_DIR | tee $TAR_DIR/log.txt
