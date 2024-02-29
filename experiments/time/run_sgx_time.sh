#!/usr/bin/env bash

OUTPUT_DIR=$1
USE_RSS=$2
export UKB_REF_PANEL_SIZE=$3
export UKB_SAMPLE_ID=4197007

MAIN_DIR=$(git rev-parse --show-toplevel)

DATASET=$MAIN_DIR/data/datasets/ukb/env_single.sh

if [[ $USE_RSS -eq 1 ]]
then
    PHASING_OPTIONS="--use-rss"
fi

HEAP_SIZE=0xC0000000
case $UKB_REF_PANEL_SIZE in
  20)
      HEAP_SIZE=0x24000000
    ;;

  40)
      HEAP_SIZE=0x30000000
    ;;

  100)
      HEAP_SIZE=0x47000000
    ;;

  200)
      HEAP_SIZE=0x80000000
    ;;

  400)
      HEAP_SIZE=0x100000000
    ;;

  *)
      HEAP_SIZE=0x100000000
    ;;
esac
echo $HEAP_SIZE

echo "[package.metadata.fortanix-sgx]
stack-size=0x200000
heap-size=$HEAP_SIZE
threads=1" > $OUTPUT_DIR/Cargo.toml

mkdir -p bin
$MAIN_DIR/experiments/sgx/build_sgx.sh bin && \
    $MAIN_DIR/experiments/sgx/run_sgx.sh bin $DATASET "$PHASING_OPTIONS" $OUTPUT_DIR $OUTPUT_DIR | tee $OUTPUT_DIR/log.txt
