#!/usr/bin/env bash

SHAPEIT4_DIR=/home/ndokmai/workspace/shapeit4/bin
REF_PANEL_DIR=$BASE_REF_PANEL_DIR/vcf
INPUT=$(find $INPUT_DIR -name *_$CHR.vcf.gz)
REF_PANEL=$(find $REF_PANEL_DIR -name ALL.chr$CHR.*.vcf.gz)
GMAP=$GMAP_DIR/chr${CHR}.b37.gmap

filename=${INPUT%.vcf.gz}
filename=${filename##*/}
log_filename=${filename}_round_$ROUND.log
filename=${filename}_phased_round_$ROUND.vcf.gz

$SHAPEIT4_DIR/shapeit4.2 \
    --input $INPUT \
    --reference $REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --thread 10 \
    --seed $RANDOM \
    --output $TAR_DIR/$filename \
    --log $TAR_DIR/$log_filename
