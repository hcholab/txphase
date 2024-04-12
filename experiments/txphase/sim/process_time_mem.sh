#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT=$OUTPUT_DIR/time_mem.txt

echo "ref_panel_size_k time_s mem_kb" > $OUTPUT

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        size="$(basename -- $dir)"
        file=$OUTPUT_DIR/$size/time_mem.txt
        echo "$size $(awk '{print $1, $2}' $file)" >> $OUTPUT
    fi
done
