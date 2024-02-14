#!/usr/bin/env bash

LITE=${1:-0}
SGX=${2:-0}

source config.sh

(cd host && cargo +nightly build $PROFILE) && \
    (cd phasing && cargo +nightly build $PROFILE $FEATURES $TARGET)
