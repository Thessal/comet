#!/bin/bash
# A simple runner script to build the Rust stdlib dynamic/shared object 
# and use LLI directly to verify that LLVM can invoke it properly.

# Exit immediately if a command exits with a non-zero status
set -e

echo "[1/2] Compiling src/stdlib/lib.rs from Rust using Cargo..."
cargo build --lib --release

echo "[2/2] Running src/stdlib/test_ts_mean.ll with lli..."
# lli works as an interpreter or JIT compiler for LLVM bitcode/IR
# The -load argument links our freshly baked stdlib .so
lli -load=target/release/libstdlib.so src/stdlib/test_ts_mean.ll

echo "[3/3] Running src/stdlib/test_add.ll with lli..."
lli -load=target/release/libstdlib.so src/stdlib/test_add.ll

echo "Success!"
