#!/usr/bin/env bash

DATA_DIR=data
source profiles/dataset/1kg.sh

./shapeit-bin/phase_common_static \
    --input $INPUT \
    --reference $VCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --thread $(nproc --all)\
    --filter-maf 0.001 \
    --output tmp/phased.bcf \
    --log tmp/log.txt

bcftools convert -O z tmp/phased.bcf -o tmp/phased.vcf.gz
