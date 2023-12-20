#!/bin/bash

source config.sh
source common.sh

REF_PANEL=$1
GMAP=$2
INPUT=$3
OUTPUT=$4

(cd host && cargo +nightly run $PROFILE $BIN_FLAGS -- $PORT $REF_PANEL $GMAP $INPUT $OUTPUT &)
(cd phasing && cargo +nightly run $PROFILE $BIN_FLAGS -- $PORT)
