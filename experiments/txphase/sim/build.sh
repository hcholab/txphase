#!/usr/bin/env bash

MAIN_DIR=$(git rev-parse --show-toplevel)

source $MAIN_DIR/config.sh

cd $MAIN_DIR
cargo +nightly build --release -p host && \
    cargo +nightly build --release -p phasing $FEATURES
