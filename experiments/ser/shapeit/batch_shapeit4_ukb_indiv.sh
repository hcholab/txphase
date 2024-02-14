#!/usr/bin/env -S bash -x

UKB_REF_PANEL_SIZE=$1
TAR_DIR=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BATCH_DIR=$TAR_DIR/${UKB_REF_PANEL_SIZE}k
CONCURRENT=$(nproc --all)

mkdir -p $BATCH_DIR

while IFS= read -r sample; do
    echo "$SCRIPT_DIR/run_shapeit_ukb_indiv.sh $sample $UKB_REF_PANEL_SIZE $BATCH_DIR"
done < "$SCRIPT_DIR/../../samples.list" | xargs -I CMD -P $CONCURRENT bash -c CMD
