#!/bin/bash

source config.sh
source common.sh

REF_PANEL=$1
REF_SITES=$2
GMAP=$3
INPUT=$4
OUTPUT=$5

(cd host && cargo +nightly run --release $BIN_FLAGS -- $REF_PANEL $REF_SITES $GMAP $INPUT $OUTPUT &)
(cd phasing && cargo +nightly run --release $BIN_FLAGS)
