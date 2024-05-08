#!/usr/bin/env bash
OUTPUT_DIR=$1

OUTPUT_TIME=$OUTPUT_DIR/time.txt

echo "sample_size ref_panel_size_k time_s" > $OUTPUT_TIME

for dir in $OUTPUT_DIR/*; do
    if [ -d "$dir" ]; then
        ./process_time.sh $dir
        file=$dir/time.txt
        size="$(basename -- $dir)"
        awk 'NR>1 {print '$size', $1, $2}' $file >> $OUTPUT_TIME
    fi
done
