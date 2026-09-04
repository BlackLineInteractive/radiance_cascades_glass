#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DIR/.." && pwd)"

echo "=== Building Metal Radiance Cascades Glass Engine ==="

# 1. Compile Metal shaders to metallib
echo "[1/2] Compiling Metal shaders..."
xcrun -sdk macosx metal -O3 -c "$ROOT_DIR/Metal/shaders/RCGlassShaders.metal" -o "$ROOT_DIR/Metal/shaders/RCGlassShaders.air"
xcrun -sdk macosx metallib "$ROOT_DIR/Metal/shaders/RCGlassShaders.air" -o "$ROOT_DIR/Metal/shaders/RCGlassShaders.metallib"
rm -f "$ROOT_DIR/Metal/shaders/RCGlassShaders.air"

# 2. Compile Cocoa / MetalKit application
echo "[2/2] Compiling application..."
clang++ -std=c++17 -O3 -Wall \
    -fobjc-arc \
    -framework Cocoa \
    -framework Metal \
    -framework MetalKit \
    -framework QuartzCore \
    -I"$ROOT_DIR/assets" \
    -I"$ROOT_DIR/common" \
    -I"$ROOT_DIR/Metal/src" \
    "$ROOT_DIR/Metal/src/main.mm" \
    -o "$ROOT_DIR/Metal/rc_glass_app"

echo "Build complete: Metal/rc_glass_app"
