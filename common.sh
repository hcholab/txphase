

RUSTFLAGS="-Ctarget-cpu=native -Ctarget-feature=+aes,+avx,+avx2,+sse2,+sse4.1,+ssse3"
export RUSTFLAGS="$RUSTFLAGS"

if [[ $LITE -eq 1 ]]
then
    BIN_FLAGS="--no-default-features"
fi
