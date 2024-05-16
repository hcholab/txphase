#!/usr/bin/env bash

OUTPUT_DIR=$1

MAIN_DIR=$(git rev-parse --show-toplevel)
source $MAIN_DIR/config.sh
source $MAIN_DIR/data/datasets/hrc/env.sh

N_REPEATS=100
for ((n=0;n<$N_REPEATS;n++)); do
    OUTPUT=$OUTPUT_DIR/phased_$n.bcf
    echo "$MAIN_DIR/shapeit-bin/phase_common_static \
            --input $(realpath $INPUT_SIZE) \
            --reference $(realpath $BCF_REF_PANEL) \
            --map $(realpath $GMAP) \
            --region $CHR \
            --seed $RANDOM \
            --output $OUTPUT_DIR/phased_$n.bcf && \
            bcftools convert -O b $OUTPUT_DIR/phased_$n.bcf -o $OUTPUT_DIR/phased_$n.vcf.gz"
done | xargs -P $(nproc --all) -I {} bash -c {}



