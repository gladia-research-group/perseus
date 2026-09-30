#!/bin/bash
# Switch between n32 and n64 deps + build dirs
# Usage: source scripts/switch_deps.sh n32   (or n64)

case "${1:-}" in
    n32|32)
        echo "Switching to NATIVEINT=32"
        export DEPS_DIR="$PWD/deps_n32"
        export NATIVE_SIZE=32
        export BUILD_DIR="$PWD/build_n32"
        ;;
    n64|64)
        echo "Switching to NATIVEINT=64"
        export DEPS_DIR="$PWD/deps_n64"
        export NATIVE_SIZE=64
        export BUILD_DIR="$PWD/build_n64"
        ;;
    *)
        echo "Usage: source scripts/switch_deps.sh n32|n64"
        echo "Currently:"
        echo "  DEPS_DIR=${DEPS_DIR:-<not set>}"
        echo "  NATIVE_SIZE=${NATIVE_SIZE:-<not set>}"
        echo "  BUILD_DIR=${BUILD_DIR:-<not set>}"
        return 1
        ;;
esac
echo "  DEPS_DIR=$DEPS_DIR"
echo "  NATIVE_SIZE=$NATIVE_SIZE"
echo "  BUILD_DIR=$BUILD_DIR"
