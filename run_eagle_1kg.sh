#!/usr/bin/env bash

source data/datasets/1kg/env.sh

./eagle-bin/eagle \
    --geneticMapFile data/gmaps/genetic_map_hg19_withX_eagle2.txt \
    --outPrefix tmp/phased \
    --numThreads $(nproc --all) \
    --vcfRef $BCF_REF_PANEL \
    --vcfTarget $INPUT_SIZE \
    --noImpMissing \
    --vcfOutFormat b \
    --chrom $CHR

bcftools convert -O z tmp/phased.bcf -o tmp/phased.vcf.gz
