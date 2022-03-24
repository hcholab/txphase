#!/usr/bin/env bash

TEST=$1
REF1=$2
REF2=$3
TRIO=$4

MERGED_NAME=.switch_rate_merged.vcf.gz
rm $MERGED_NAME 2> /dev/null
bcftools index -t $TEST > /dev/null 2>&1
bcftools merge -m none -o $MERGED_NAME -O z $TEST $REF1 $REF2 > /dev/null 2>&1

#Child, nTested, nMendelian Errors, nSwitch, nSwitch (%)
bcftools +trio-switch-rate $MERGED_NAME -- -p $TRIO | \
    awk '{if($1=="TRIO"){ print $4, $5, $6, $7, $8 }}'
