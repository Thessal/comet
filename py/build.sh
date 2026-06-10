#!/bin/bash

# I recompiled the Rust py module to dynamically link strictly to the exact same libtorch.so that your Python .venv uses.
source /home/jongkook90/comet/.venv/bin/activate
unset LIBTORCH
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo clean -p torch-sys
cargo clean -p tch
cargo build --release
cp ../target/release/libcomet_env.so ./comet_env.so
ldd comet_env.so > ldd_output.txt
python test.py > py_output.txt 2>&1