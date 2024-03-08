#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT_MEM=$OUTPUT_DIR/mem.txt

echo "ref_panel_size_k mem_kb" > $OUTPUT_MEM

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        size="$(basename -- $dir)"
        file=$OUTPUT_DIR/$size/time_mem.txt
        awk '{print '$size', $2}' $file >> $OUTPUT_MEM
    fi
done
