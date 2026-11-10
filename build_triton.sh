#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLVM_BUILD_DIR=/home/yaoshi16ultra/workspace/llvm-project/build
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
export TRITON_BUILD_PROTON=OFF
export MAX_JOBS=16
export TRITON_PLUGIN_DIRS="${SCRIPT_DIR}/third_party/privateuse"
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF -DMLIR_ENABLE_DOCS=OFF"

if [ "$1" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf ${SCRIPT_DIR}/build
    cd ${SCRIPT_DIR} && pip uninstall -y triton
    echo "Clean complete."
    exit 0
fi

if [ "$1" = "wheel" ]; then
    shift
    WHEEL_DIR="${SCRIPT_DIR}/dist"
    mkdir -p "${WHEEL_DIR}"
    echo "Building wheel into ${WHEEL_DIR}..."
    cd ${SCRIPT_DIR} && pip wheel . --no-index --no-deps --no-build-isolation --wheel-dir "${WHEEL_DIR}" "$@" || exit 1
    echo "Wheel complete:"
    ls -lh "${WHEEL_DIR}"/triton-*.whl
    exit 0
fi

cd ${SCRIPT_DIR} && pip install -vvv -e . --no-index --no-deps --no-build-isolation
