#!/usr/bin/env bash

SAMPLES=$1
N_SAMPLES=$2
OUTPUT_DIR=$3
LIST=${4}

OUTPUT_BASE=$(basename $SAMPLES)
OUTPUT=$OUTPUT_DIR/${OUTPUT_BASE%%.*}_${N_SAMPLES}.bcf



#bcftools query -l $SAMPLES > .list.1
#REF_N_SAMPLES=$(wc -l $LIST)

mkdir -p $OUTPUT_DIR

if [ ! -f $OUTPUT ]; then
    #if (($N_SAMPLES < $REF_N_SAMPLES)); then
        #shuf -n $N_SAMPLES $LIST > .list.2
        head -n $N_SAMPLES $LIST > .list.2
        bcftools view -S .list.2 -o $OUTPUT -O b $SAMPLES
    #elif (($N_SAMPLES > $REF_N_SAMPLES)); then
        #N_LOOPS=$(expr $N_SAMPLES / $REF_N_SAMPLES)
        #REM=$(expr $N_SAMPLES % $REF_N_SAMPLES)
        #for i in $(seq 1 $N_LOOPS); do
            #awk '{print $1"'_$i'"}' .list.1 > .list.2
            #bcftools reheader --samples .list.2 -o $OUTPUT.$i $SAMPLES
            #bcftools index $OUTPUT.$i
            #files[$(expr $i - 1)]=$OUTPUT.$i
        #done
        #if (( $REM > 1 )); then
            #LAST_LOOP=$(expr $N_LOOPS + 1)
            #shuf -e -n $REM $LIST> .list.txt
            #bcftools view -S .list.txt -o $OUTPUT.$LAST_LOOP.tmp -O b $SAMPLES
            #cp .list.txt .list.txt.2
            #awk '{print $1"'_$LAST_LOOP'"}' .list.txt.2 > .list.txt
            #bcftools reheader --samples .list.txt -o $OUTPUT.$LAST_LOOP $OUTPUT.$LAST_LOOP.tmp
            #bcftools index $OUTPUT.$LAST_LOOP
            #files[$(expr $N_LOOPS)]=$OUTPUT.$LAST_LOOP

            #bcftools merge -o $OUTPUT -O b ${files[@]}

            #rm -f .list.txt .list.txt.2 $OUTPUT.*
        #fi
    #else
        #cp $SAMPLES $OUTPUT
    #fi
    bcftools index $OUTPUT
else
    echo "$OUTPUT already exists."
fi

rm -f .list.*


