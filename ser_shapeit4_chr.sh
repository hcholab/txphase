#!/bin/bash

CHR=$1
PHASED=$2

BASE_DIR=$(realpath data/data/giab)
PARENT1_DIR=$BASE_DIR/HG002/parent1/chr
PARENT2_DIR=$BASE_DIR/HG002/parent2/chr
TRIO=$BASE_DIR/trio.ped

parent1=$(find $PARENT1_DIR -name *_${CHR}.vcf.gz)
parent2=$(find $PARENT2_DIR -name *_${CHR}.vcf.gz)
./benchmark/switch-error-rate.sh $PHASED $parent1 $parent2 $TRIO
