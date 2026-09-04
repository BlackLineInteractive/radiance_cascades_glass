#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

#if defined(__APPLE__) && !defined(TEAPOT_DATA_NO_SIMD)
#include <simd/simd.h>
using TeapotVec4 = simd::float4;
using TeapotVec3 = simd::float3;
#define TEAPOT_MAKE_VEC3(x, y, z) simd::make_float3(x, y, z)
#else
struct alignas(16) TeapotVec4 {
    float x, y, z, w;
    TeapotVec4() : x(0), y(0), z(0), w(0) {}
    TeapotVec4(float x, float y, float z, float w = 0.0f) : x(x), y(y), z(z), w(w) {}
};

struct TeapotVec3 {
    float x, y, z;
    TeapotVec3() : x(0), y(0), z(0) {}
    TeapotVec3(float x, float y, float z) : x(x), y(y), z(z) {}
};
#define TEAPOT_MAKE_VEC3(x, y, z) TeapotVec3(x, y, z)
#endif

struct alignas(16) GPUBVHNode {
    TeapotVec4 bmin;
    TeapotVec4 bmax;
    int32_t leftChild;
    int32_t rightChild;
    int32_t pad[2];
};

struct alignas(16) GPUTriangle {
    TeapotVec4 v0;
    TeapotVec4 v1;
    TeapotVec4 v2;
    TeapotVec4 n0;
    TeapotVec4 n1;
    TeapotVec4 n2;
};

static_assert(sizeof(GPUBVHNode) == 48, "GPUBVHNode must be 48 bytes");
static_assert(sizeof(GPUTriangle) == 96, "GPUTriangle must be 96 bytes");

struct TeapotMesh {
    std::vector<GPUBVHNode> nodes;
    std::vector<GPUTriangle> triangles;
    TeapotVec3 boundsMin;
    TeapotVec3 boundsMax;
    TeapotVec3 center;

    bool loadFromBinary(const std::string &filePath) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[TeapotMesh] Warning: Unable to open " << filePath << "\n";
            return false;
        }

        uint32_t numNodes = 0;
        uint32_t numTris = 0;
        file.read(reinterpret_cast<char *>(&numNodes), sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&numTris), sizeof(uint32_t));

        if (numNodes == 0 || numTris == 0 || numNodes > 100000 || numTris > 100000) {
            std::cerr << "[TeapotMesh] Invalid header in " << filePath << "\n";
            return false;
        }

        nodes.resize(numNodes);
        triangles.resize(numTris);

        file.read(reinterpret_cast<char *>(nodes.data()), numNodes * sizeof(GPUBVHNode));
        file.read(reinterpret_cast<char *>(triangles.data()), numTris * sizeof(GPUTriangle));

        if (!nodes.empty()) {
            boundsMin = TEAPOT_MAKE_VEC3(nodes[0].bmin.x, nodes[0].bmin.y, nodes[0].bmin.z);
            boundsMax = TEAPOT_MAKE_VEC3(nodes[0].bmax.x, nodes[0].bmax.y, nodes[0].bmax.z);
            center = TEAPOT_MAKE_VEC3(
                (boundsMin.x + boundsMax.x) * 0.5f,
                (boundsMin.y + boundsMax.y) * 0.5f,
                (boundsMin.z + boundsMax.z) * 0.5f
            );
        }

        std::cout << "[TeapotMesh] Loaded " << numTris << " triangles, "
                  << numNodes << " BVH nodes from " << filePath << "\n";
        return true;
    }
};
