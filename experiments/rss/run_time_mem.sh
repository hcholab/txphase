#!/usr/bin/env bash

TAR_DIR=$1
USE_RSS=$2
PBWT_DEPTH=$3

MAIN_DIR=$(git rev-parse --show-toplevel)

if [[ $USE_RSS -eq 1 ]]
then
    PHASING_OPTIONS="--use-rss --max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100 --pbwt-depth $PBWT_DEPTH"
else
    PHASING_OPTIONS="--max-m3vcf-unique-haps=200 --min-m3vcf-unique-haps=100 --pbwt-depth $PBWT_DEPTH"
fi

$MAIN_DIR/experiments/nosgx/build.sh bin
$MAIN_DIR/experiments/nosgx/run.sh bin $MAIN_DIR/profiles/dataset/1kg.sh "$PHASING_OPTIONS" $TAR_DIR $TAR_DIR | tee $TAR_DIR/log.txt
