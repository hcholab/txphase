#!/usr/bin/env bash

TARGET_DIR=$1

LITE=0
SGX=1

MAIN_DIR=$(git rev-parse --show-toplevel)

PHASING=$MAIN_DIR/phasing/target/x86_64-fortanix-unknown-sgx/release/phasing
HOST=$MAIN_DIR/host/target/release/host

(
cd $MAIN_DIR && ./build.sh $LITE $SGX
) && (
cp $PHASING $TARGET_DIR && cp $HOST $TARGET_DIR
)
