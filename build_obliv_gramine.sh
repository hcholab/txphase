#!/usr/bin/env bash

source config.sh

(cd host && cargo +nightly build $PROFILE) && \
    (cd phasing && make SGX=1 EDMM=1 RUSTFLAGS="$RUSTFLAGS" PROFILE="$PROFILE")

