#!/bin/bash

source config.sh
source common.sh

REF_PANEL=$1
REF_SITES=$2
GMAP=$3
INPUT=$4
OUTPUT=$5

(cd host && cargo +nightly run $PROFILE $BIN_FLAGS -- $PORT $REF_PANEL $REF_SITES $GMAP $INPUT $OUTPUT &)
(cd phasing && cargo +nightly run $PROFILE --features leak-resist-new $BIN_FLAGS -- $PORT)
