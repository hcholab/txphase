#!/bin/bash

export CHR=$1
export TAR_DIR=$(realpath $2)
export RANDOM=$3

BASE_DIR=$(realpath data/data)
export BASE_REF_PANEL_DIR=$BASE_DIR/1kg
export INPUT_DIR=$BASE_DIR/giab/HG002/chr
export GMAP_DIR=$BASE_DIR/maps
export ROUND=1

./benchmark/shapeit4.sh
