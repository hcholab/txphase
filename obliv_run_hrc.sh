#!/bin/bash

CHR=20
BASE_DIR=$(realpath data/data)
REF_PANEL=$BASE_DIR/hrc_chr20.m3vcf.gz
REF_SITES=$BASE_DIR/hrc_chr${CHR}.csv
GMAP=$BASE_DIR/maps/chr${CHR}.b37.gmap
INPUT=$(find $BASE_DIR/giab/HG002/chr -name "*_${CHR}.vcf.gz")
OUTPUT=chr${CHR}_phased.vcf.gz

./obliv_run.sh $REF_PANEL $REF_SITES $GMAP $INPUT $OUTPUT
