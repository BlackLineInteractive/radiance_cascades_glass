#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DIR/.." && pwd)"

echo "=== Building OpenGL 4.3+ Radiance Cascades Glass Engine ==="

# Check GLFW and GLM
GLFW_INC="-I/usr/local/include"
GLFW_LIB="-L/usr/local/lib -lglfw"
if [ -d "/opt/homebrew/include" ]; then
    GLFW_INC="$GLFW_INC -I/opt/homebrew/include"
    GLFW_LIB="$GLFW_LIB -L/opt/homebrew/lib"
fi

PLATFORM_FLAGS=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM_FLAGS="-framework OpenGL"
else
    PLATFORM_FLAGS="-lGL"
fi

clang++ -std=c++17 -O3 -Wall \
    -I"$ROOT_DIR/assets" \
    -I"$ROOT_DIR/common" \
    -I"$ROOT_DIR/OpenGL/src" \
    $GLFW_INC \
    $GLFW_LIB \
    $PLATFORM_FLAGS \
    "$ROOT_DIR/OpenGL/src/main.cpp" \
    "$ROOT_DIR/OpenGL/src/gl_loader.cpp" \
    -o "$ROOT_DIR/OpenGL/rc_glass_gl"

echo "Build complete: OpenGL/rc_glass_gl"
