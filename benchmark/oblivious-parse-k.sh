#!/bin/bash

LOG_FILE_DIR=$1

LOG_FILES=$(ls -tr $LOG_FILE_DIR/*.log)

echo "chromosome, round, b1, b2, b3, b4, b5, p1, b6, p2, b7, p2, m1, m2, m3, m4, m5,"
for file in $LOG_FILES
do
    chr=${file##*benchmark_}
    chr=${chr%%_round*}
    round=${file##*round_}
    round=${round%%.log}
    printf "$chr, $round, "
    awk '{if($1=="K:") { printf "%s, ", $2 }}' $file
    echo ""
done
