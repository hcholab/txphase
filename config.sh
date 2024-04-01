PROFILE="--release"
#FEATURES="--features benchmarking"
if [[ $LITE -eq 1 ]]
then
    FEATURES += "--no-default-features --features lite"
fi

if [[ $SGX -eq 1 ]]
then
    TARGET="--target x86_64-fortanix-unknown-sgx"
fi

export RUSTFLAGS="-Ctarget-cpu=native -Ctarget-feature=+aes,+avx,+avx2,+sse2,+sse4.1,+ssse3"
