#!/usr/bin/env bash

source data/datasets/1kg/env.sh

./shapeit-bin/phase_common_static \
    --input $INPUT_SIZE \
    --reference $BCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --thread $(nproc --all) \
    --output tmp/phased.bcf \
    --log tmp/log.txt

bcftools convert -O z tmp/phased.bcf -o tmp/phased.vcf.gz
