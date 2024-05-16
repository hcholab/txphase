#!/usr/bin/env bash

SRC_DIR=$1
SER_FILE=$2
OUT_DIR=tmp

mkdir -p $OUT_DIR

MAIN_DIR=$(git rev-parse --show-toplevel)
source $MAIN_DIR/data/datasets/1kg/env.sh

find $SRC_DIR -name "phased_*.vcf.gz" | xargs -P $(nproc --all) -I {} bash -c "$MAIN_DIR/scripts/switch-error-rate.sh {} $PARENTS $TRIO $OUT_DIR"

for file in $OUT_DIR/*.txt; do
    awk '$1 == "TRIO" && $2 == "HG003" && $3 == "HG004" && $4 == "HG002" { print $8 }' $file
done > $SER_FILE

rm -rf tmp
