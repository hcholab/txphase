#!/usr/bin/env bash

export UKB_REF_PANEL_SIZE=$1
#export UKB_SAMPLE_ID=4197007
export UKB_SAMPLE_ID=2053761
./run.sh 0 0 data profiles/dataset/ukb_single.sh profiles/phasing/rss-ukb20.sh
