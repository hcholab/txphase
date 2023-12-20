#!/bin/bash

CHR=$1
BASE_DIR=$(realpath data/data)
REF_PANEL_BASE_DIR=$BASE_DIR/1kg
REF_PANEL=$(find $REF_PANEL_BASE_DIR/m3vcf -name "${CHR}.*.m3vcf.gz")
GMAP=$BASE_DIR/maps/chr${CHR}.b37.gmap
INPUT=$(find $BASE_DIR/giab/HG002/chr -name "*_${CHR}.vcf.gz")
OUTPUT=chr${CHR}_phased.vcf.gz

./obliv_run.sh $REF_PANEL $GMAP $INPUT $OUTPUT
