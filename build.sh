#!/bin/bash

source config.sh
source common.sh

(cd host && cargo +nightly build $PROFILE $BIN_FLAGS)
(cd phasing && cargo +nightly build --no-default-features --features compressed-pbwt $PROFILE $BIN_FLAGS)
