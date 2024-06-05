#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT_SER=$OUTPUT_DIR/ser.txt

echo "ref_panel_size_k child_id n_tested n_switches" > $OUTPUT_SER

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        size="$(basename -- $dir)"
        file=$(find $OUTPUT_DIR/$size/ -name 'ser_*.txt' | head -n 1)
        if [ -n "${file}" ]; then
            tail -n +5 $file | head -n -2 | awk '{print '$size', $4, $5, $7}' >> $OUTPUT_SER
        fi
    fi
done
