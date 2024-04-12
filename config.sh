PORT=7777
N_WORKERS=$(nproc --all)
#FEATURES="--features benchmarking"
export RUSTFLAGS="-Ctarget-cpu=native"
