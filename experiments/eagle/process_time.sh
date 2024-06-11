#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT_TIME=$OUTPUT_DIR/time.txt

echo "ref_panel_size_k time_s" > $OUTPUT_TIME

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        size="$(basename -- $dir)"
        file=$OUTPUT_DIR/$size/time.txt
        echo "$size $(awk '{print $1}' $file)" >> $OUTPUT_TIME
    fi
done
