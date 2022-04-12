#!/bin/bash

source config.sh
source common.sh

(cd host && cargo +nightly build --release $BIN_FLAGS)
(cd phasing && cargo +nightly build --release --features leak-resist-new $BIN_FLAGS)
