#!/bin/bash

source config.sh
source common.sh

(cd host && cargo +nightly build $PROFILE  $BIN_FLAGS)
(cd phasing && cargo +nightly build --target x86_64-fortanix-unknown-sgx $PROFILE --features leak-resist-new $BIN_FLAGS)
