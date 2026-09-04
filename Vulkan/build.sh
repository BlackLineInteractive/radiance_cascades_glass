#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DIR/.." && pwd)"

echo "=== Building Vulkan 1.2+ Radiance Cascades Glass Engine ==="

# 1. Compile GLSL compute shaders to SPIR-V
echo "[1/2] Compiling Vulkan GLSL compute shaders to SPIR-V..."
glslangValidator -V "$ROOT_DIR/Vulkan/shaders/radiance_cascades.comp" -o "$ROOT_DIR/Vulkan/shaders/radiance_cascades.spv"
glslangValidator -V "$ROOT_DIR/Vulkan/shaders/filter_atlas.comp" -o "$ROOT_DIR/Vulkan/shaders/filter_atlas.spv"
glslangValidator -V "$ROOT_DIR/Vulkan/shaders/caustics_generate.comp" -o "$ROOT_DIR/Vulkan/shaders/caustics_generate.spv"
glslangValidator -V "$ROOT_DIR/Vulkan/shaders/caustics_filter.comp" -o "$ROOT_DIR/Vulkan/shaders/caustics_filter.spv"
glslangValidator -V "$ROOT_DIR/Vulkan/shaders/render_scene.comp" -o "$ROOT_DIR/Vulkan/shaders/render_scene.spv"

# 2. Compile host application
echo "[2/2] Compiling Vulkan application..."
VK_INC="-I/usr/local/include"
VK_LIB="-L/usr/local/lib -lvulkan -lglfw"
if [ -d "/opt/homebrew/include" ]; then
    VK_INC="$VK_INC -I/opt/homebrew/include"
    VK_LIB="$VK_LIB -L/opt/homebrew/lib"
fi

clang++ -std=c++17 -O3 -Wall \
    -I"$ROOT_DIR/assets" \
    -I"$ROOT_DIR/common" \
    -I"$ROOT_DIR/Vulkan/src" \
    $VK_INC \
    $VK_LIB \
    "$ROOT_DIR/Vulkan/src/main.cpp" \
    -o "$ROOT_DIR/Vulkan/rc_glass_vk"

echo "Build complete: Vulkan/rc_glass_vk"
