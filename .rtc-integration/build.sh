#!/bin/bash
# Build the lacemodelica executable of the rtc-integration branch: build/lacemodelica. rtc-tools
# finds it on the PATH (.rtc-integration/env.sh of rtc-tools). The ANTLR runtime and ONNX come
# from an existing lacemodelica build (DEPS), to save the download and build.
set -euxo pipefail
cd "$(dirname "$0")/.."
DEPS=${DEPS:-/home/yacoda/programs/lacemodelica/build}
${CLEAN:+rm -rf build}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLACEMODELICA_BUILD_EXE=ON \
  -DFETCHCONTENT_SOURCE_DIR_ANTLR4_RUNTIME="$DEPS/_deps/antlr4_runtime-src" \
  -DCMAKE_PREFIX_PATH="$DEPS/external_projects"
cmake --build build --parallel "${JOBS:-8}" --target lacemodelica_exe
