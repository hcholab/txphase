#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2

MAIN_DIR=$(git rev-parse --show-toplevel)
source $MAIN_DIR/data/datasets/ukb/env_all.sh

SIZE_DIR=$OUTPUT_DIR/$UKB_REF_PANEL_SIZE

mkdir -p $SIZE_DIR

/usr/bin/time -f "%e %M" -o $SIZE_DIR/time_mem.txt $MAIN_DIR/eagle-bin/eagle \
    --geneticMapFile $MAIN_DIR/data/gmaps/genetic_map_hg19_withX_eagle2.txt \
    --outPrefix $SIZE_DIR/phased \
    --numThreads $(nproc --all) \
    --vcfRef $BCF_REF_PANEL \
    --vcfTarget $INPUT_SER \
    --noImpMissing \
    --vcfOutFormat b \
    --chrom $CHR | \
    tee $SIZE_DIR/log.txt && \
$MAIN_DIR/scripts/switch-error-rate.sh $SIZE_DIR/phased.bcf $PARENTS $TRIO $SIZE_DIR
