#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2
export UKB_INPUT_SIZE=$3

MAIN_DIR=$(git rev-parse --show-toplevel)
source $MAIN_DIR/data/datasets/ukb/env_all.sh

SIZE_DIR=$OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE
mkdir -p $SIZE_DIR
SIZE_DIR=$(realpath $OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE)

N_CPUS=$(nproc --all)

cd $MAIN_DIR/eagle-bin
/usr/bin/time -f "%e" -o $SIZE_DIR/time.txt gramine-sgx $MAIN_DIR/eagle-bin/eagle \
    --geneticMapFile $MAIN_DIR/data/gmaps/genetic_map_hg19_withX_eagle2.txt \
    --outPrefix $SIZE_DIR/phased \
    --numThreads $(nproc --all) \
    --vcfRef $BCF_REF_PANEL \
    --vcfTarget $INPUT_SIZE \
    --noImpMissing \
    --vcfOutFormat b \
    --chrom $CHR | \
    tee $SIZE_DIR/log.txt
