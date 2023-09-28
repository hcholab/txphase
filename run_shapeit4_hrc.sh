#!/bin/bash

export CHR=$1
export TAR_DIR=$(realpath $2)
export RANDOM=$3

BASE_DIR=$(realpath data/data)
INPUT_DIR=$BASE_DIR/giab/HG002/chr
GMAP_DIR=$BASE_DIR/maps
ROUND=1

SHAPEIT4_DIR=/home/ndokmai/app/shapeit4/bin
REF_PANEL_DIR=$BASE_REF_PANEL_DIR/vcf
INPUT=$(find $INPUT_DIR -name *_$CHR.vcf.gz)
REF_PANEL=$BASE_DIR/HRC.r1-1.EGA.GRCh37.chr20.haplotypes.GeneImp.vcf.gz
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
    --thread 4 \
    --seed $RANDOM \
    --output $TAR_DIR/$filename \
    --log $TAR_DIR/$log_filename

