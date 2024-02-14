#!/usr/bin/env bash

TEST=$1
REF=$2
TRIO=$3
TAR_DIR=$4

if test -z "$TAR_DIR"
then
    TAR_DIR=.
fi

MERGED=$TAR_DIR/.switch_rate_merged.vcf.gz
SER_OUT=$TAR_DIR/ser.txt

rm -f $MERGED
bcftools index -f $TEST
bcftools merge -m none -o $MERGED -O z $TEST $REF

bcftools +trio-switch-rate $MERGED -- -p $TRIO | tee $SER_OUT
rm -f $MERGED
