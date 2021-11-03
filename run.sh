#!/bin/bash

source config.sh
source common.sh

(cd host && cargo +nightly run --release $BIN_FLAGS &)
(cd phasing && cargo +nightly run --release $BIN_FLAGS)
