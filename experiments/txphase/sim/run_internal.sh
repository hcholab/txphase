#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2
export UKB_INPUT_SIZE=$3

OUTPUT_DIR_SIZE=$(realpath $OUTPUT_DIR/$UKB_INPUT_SIZE/$UKB_REF_PANEL_SIZE)

MAIN_DIR=$(git rev-parse --show-toplevel)

source $MAIN_DIR/config.sh
source $MAIN_DIR/data/datasets/ukb/env_all.sh

mkdir -p $OUTPUT_DIR_SIZE

N_WORKERS=$(($UKB_INPUT_SIZE < $N_WORKERS ? $UKB_INPUT_SIZE: $N_WORKERS))

HOST_OPTIONS="\
    --worker-port-base $PORT \
    --n-workers $N_WORKERS \
    --ref-panel $(realpath $M3VCF_REF_PANEL) \
    --genetic-map $(realpath $GMAP) \
    --input $(realpath $INPUT_SIZE) \
    --output $(realpath $OUTPUT_DIR_SIZE/phased.vcf.gz)"

killall host >/dev/null 2>&1

(cd $MAIN_DIR
(for worker_id in $(seq 0 $(($N_WORKERS - 1)))
do
    /usr/bin/time -f "%M" -o $OUTPUT_DIR_SIZE/worker_${worker_id}_mem.txt target/release/phasing --host-port $(($PORT + $worker_id)) | tee $OUTPUT_DIR_SIZE/worker_${worker_id}.log &
    pid[$worker_id]=$!
done
trap "kill $(printf " %d" "${pid[@]}"); exit 1" INT
/usr/bin/time -f "%M" -o $OUTPUT_DIR_SIZE/host_mem.txt target/release/host $HOST_OPTIONS | tee $OUTPUT_DIR_SIZE/host.log
))
