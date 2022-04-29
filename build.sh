#!/bin/bash

source config.sh
source common.sh

(cd host && cargo +nightly build $PROFILE $BIN_FLAGS)
(cd phasing && cargo +nightly build $PROFILE $BIN_FLAGS)
