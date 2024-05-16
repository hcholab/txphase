#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT=$OUTPUT_DIR/time_mem.txt

echo "ref_panel_size_k time_s mem_kb" > $OUTPUT

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        size="$(basename -- $dir)"
        time_s=$(cat $OUTPUT_DIR/$size/time.txt)
        mem_kb=$(find $OUTPUT_DIR/$size -type f -regex '.*/\(host_mem\.txt\|worker_.*_mem\.txt\)$' -exec cat {} \; | awk '{s+=$1} END {print s}')
        echo "$size $time_s $mem_kb" >> $OUTPUT
    fi
done
