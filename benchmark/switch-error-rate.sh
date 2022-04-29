#!/usr/bin/env bash

TEST=$1
REF1=$2
REF2=$3
TRIO=$4

if test -z "$TAR_DIR"
then
    TAR_DIR=.
fi

MERGED=$TAR_DIR/.switch_rate_merged.vcf.gz

rm $MERGED_NAME 2> /dev/null
bcftools index -t $TEST > /dev/null 2>&1
bcftools merge -m none -o $MERGED -O z $TEST $REF1 $REF2 > /dev/null 2>&1

#Child, nTested, nMendelian Errors, nSwitch, nSwitch (%)
bcftools +trio-switch-rate $MERGED -- -p $TRIO | \
    awk '{if($1=="TRIO"){ print $4, $5, $6, $7, $8 }}'
