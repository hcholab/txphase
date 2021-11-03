#!/bin/bash

source config.sh
source common.sh

echo "$BIN_FLAGS"

(cd host && cargo +nightly build --release $BIN_FLAGS)
(cd phasing && cargo +nightly build --release $BIN_FLAGS)
