#!/usr/bin/env bash

TARGET_DIR=$1

MAIN_DIR=$(git rev-parse --show-toplevel)
GRAMINE_DIR=$MAIN_DIR/experiments/sgx/gramine
HOST=$MAIN_DIR/host/target/release/host

source $MAIN_DIR/config.sh

(cd $MAIN_DIR/host && cargo +nightly build $PROFILE) && \
    (cp $HOST $TARGET_DIR) && \
    (cd $TARGET_DIR && make -C $GRAMINE_DIR SGX=1 DEBUG=0 EDMM=1 PROFILE="$PROFILE")
