#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT=$OUTPUT_DIR/time_mem.txt

echo "sample_size ref_panel_size_k time_s mem_kb" > $OUTPUT

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        ./process_time_mem.sh $dir
        file=$dir/time_mem.txt
        size="$(basename -- $dir)"
        awk 'NR>1 {print '$size', $1, $2, $3}' $file >> $OUTPUT
    fi
done
