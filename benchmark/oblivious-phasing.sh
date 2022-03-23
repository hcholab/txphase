#!/usr/bin/env bash

REF_PANEL_DIR=$BASE_REF_PANEL_DIR/m3vcf
REF_PANEL=$(find $REF_PANEL_DIR -name "${CHR}.*.m3vcf.gz")
INPUT=$(find $INPUT_DIR -name *_$CHR.vcf.gz)
REF_SITES=$BASE_REF_PANEL_DIR/sites/chr${CHR}.csv
GMAP=$GMAP_DIR/chr${CHR}.b37.gmap

filename=${INPUT%.vcf.gz}
filename=${filename##*/}
log_filename=${filename}_round_$ROUND.log
filename=${filename}_phased_round_$ROUND.vcf.gz

cd ..
./run.sh $REF_PANEL $REF_SITES $GMAP $INPUT $TAR_DIR/$filename 2>&1 | tee $TAR_DIR/$log_filename
