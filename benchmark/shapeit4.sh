#!/usr/bin/env bash

SHAPEIT5_DIR=/home/nd549/workspace/shapeit5/
REF_PANEL_DIR=$BASE_REF_PANEL_DIR/vcf
INPUT=$(find $INPUT_DIR -name *_$CHR.ac.vcf.gz)
REF_PANEL=$(find $REF_PANEL_DIR -name ALL.chr$CHR.*.ac.giab.norare.vcf.gz)
GMAP=$GMAP_DIR/chr${CHR}.b37.gmap

filename=${INPUT%.vcf.gz}
filename=${filename##*/}
log_filename=${filename}_round_$ROUND.log
filename=${filename}_phased_round_$ROUND.bcf

$SHAPEIT5_DIR/phase_common_static \
    --input $INPUT \
    --reference $REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --thread 20 \
    --seed $RANDOM \
    --output $TAR_DIR/$filename \
    --log $TAR_DIR/$log_filename
