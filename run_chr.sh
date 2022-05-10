#!/bin/bash

RUN=$1
CHR=$2
BASE_DIR=$(realpath data/data)
REF_PANEL_BASE_DIR=$BASE_DIR/1kg
REF_PANEL=$(find $REF_PANEL_BASE_DIR/m3vcf -name "${CHR}.*.m3vcf.gz")
REF_SITES=$REF_PANEL_BASE_DIR/sites/chr${CHR}.csv
GMAP=$BASE_DIR/maps/chr${CHR}.b37.gmap
INPUT=$(find $BASE_DIR/giab/HG002/chr -name "*_${CHR}.vcf.gz")
OUTPUT=chr${CHR}_phased.vcf.gz

./$RUN $REF_PANEL $REF_SITES $GMAP $INPUT $OUTPUT
