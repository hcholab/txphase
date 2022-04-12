#!/usr/bin/env bash

COMMAND=$1
CPU=$2
export TAR_DIR=$(realpath $3)

BASE_DIR=$(realpath ../data/data)
BASE_INPUT_DIR=$BASE_DIR/giab/HG002
export BASE_REF_PANEL_DIR=$BASE_DIR/1kg

export GMAP_DIR=$BASE_DIR/maps
export SITES_DIR=$BASE_REF_PANEL_DIR/sites
export INPUT_DIR=$BASE_INPUT_DIR/chr
PARENT1_DIR=$BASE_INPUT_DIR/parent1/chr
PARENT2_DIR=$BASE_INPUT_DIR/parent2/chr
TRIO=$BASE_DIR/giab/trio.ped

NROUNDS=10
OUTPUT_SUMMARY=$TAR_DIR/summary.txt

#CHR_LIST="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
CHR_LIST="20"
C='\033[1;33m' # Color
NC='\033[0m' # No Color

echo "chromosome round child n_tested n_mendelian_errors n_switch n_switch_percentage time_sec" > $OUTPUT_SUMMARY

for chr in $CHR_LIST
do
    export CHR=$chr
    echo -e "${C}== Processing chromosome $chr ==${NC}"
    for round in $(seq 1 $NROUNDS)
    do
        export ROUND=$round
        echo -e "${C}Round $round started${NC}"
        printf "$chr $round " >> $OUTPUT_SUMMARY
        /usr/bin/time -f %e -o .time_log taskset -c $CPU bash $COMMAND
        echo -e "${C}Round $round ended${NC}"
        phased=$(find $TAR_DIR -name *_${chr}_phased_round_${round}.vcf.gz)
        parent1=$(find $PARENT1_DIR -name *_${chr}.vcf.gz)
        parent2=$(find $PARENT2_DIR -name *_${chr}.vcf.gz)
        echo -n "$(./switch-error-rate.sh $phased $parent1 $parent2 $TRIO)" >> $OUTPUT_SUMMARY
        echo " $(cat .time_log)" >> $OUTPUT_SUMMARY
    done
    echo -e "${C}== Done processing chromosome $chr ==${NC}"
done
