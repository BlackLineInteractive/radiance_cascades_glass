#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <IOKit/IOKitLib.h>
#include <simd/simd.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cstring>

#include "TeapotData.h"
#include "Camera.h"

struct CascadeLevelParams {
    float tMin;
    float tMax;
    uint32_t raysThisLevel;
    uint32_t raysUpperLevel;          // 0 marks the terminal (farthest) cascade.
    uint32_t probesPerAxisThisLevel;
    uint32_t probesPerAxisUpperLevel; // unused when raysUpperLevel == 0
    float skyBoost;
};

// Cascade N covers [kCascadeRanges[N], kCascadeRanges[N+1]] along a ray, gathered
// over a probesPerAxis x probesPerAxis grid with `rays` directions per probe.
// Ray count quadruples per level (matching the 4x drop in probe density) so the
// total ray budget - probes^2 * rays - is the same at every level.
struct CascadeLevelSpec {
    uint32_t probesPerAxis;
    uint32_t rays;
};
static constexpr float kCascadeRanges[5] = { 0.005f, 0.25f, 0.80f, 2.50f, 100.0f };
static constexpr CascadeLevelSpec kCascadeLevelSpecs[4] = {
    { 64, 16 },
    { 32, 64 },
    { 16, 256 },
    { 8, 1024 },
};

struct GlassUniforms {
    simd::float4x4 viewInverse;
    simd::float4x4 projectionInverse;
    simd::float3 cameraPosition;
    float time;

    simd::float3 sunDirection;
    float sunIntensity;
    simd::float3 sunColor;
    float ambientIntensity;

    float glassIor;
    float glassDispersion;
    float glassRoughness;
    float glassAbsorption;

    uint32_t renderMode;
    uint32_t width;
    uint32_t height;
    uint32_t frameIndex;

    uint32_t numTeapotNodes;
    uint32_t numTeapotTris;
    uint32_t ablationMask;
    uint32_t glassBounces;
};

// Mirrors PathTraceParams in RCGlassShaders.metal.
struct PathTraceParams {
    uint32_t samplesPerLaunch;
    uint32_t sampleBase;
    uint32_t maxDepth;
    uint32_t rrStartDepth;
    float sunAngularRadius;
    float indirectClamp;
    float exposure;
    uint32_t seedOffset;
};

// Ablation bits. Bits 0, 1, 3 and 5 are read by the shader; bits 2 and 4 are
// acted on here, since they change which passes are dispatched at all.
enum AblationBit : uint32_t {
    kAblCascadeGI    = 1u << 0,  // cascade irradiance -> flat 0.04 ambient
    kAblCaustics     = 1u << 1,  // forward-splatted floor caustics
    kAblAtlasFilter  = 1u << 2,  // 7x7 Gaussian over the probe atlas
    kAblTemporal     = 1u << 3,  // 0.70 history blend on the atlas
    kAblCascadeMerge = 1u << 4,  // 4-level hierarchy -> one flat level 0 gather
    kAblDispersion   = 1u << 5,  // per-channel IOR split
    kAblAll          = 0x3Fu
};

static bool saveTGA(const std::string &path, uint32_t width, uint32_t height, const std::vector<uint8_t> &rgba) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    uint8_t header[18] = { 0 };
    header[2] = 2;
    header[12] = width & 0xFF;
    header[13] = (width >> 8) & 0xFF;
    header[14] = height & 0xFF;
    header[15] = (height >> 8) & 0xFF;
    header[16] = 32;
    header[17] = 0x20;

    file.write(reinterpret_cast<const char *>(header), sizeof(header));

    std::vector<uint8_t> bgra(width * height * 4);
    for (size_t i = 0; i < width * height; i++) {
        bgra[i * 4 + 0] = rgba[i * 4 + 2];
        bgra[i * 4 + 1] = rgba[i * 4 + 1];
        bgra[i * 4 + 2] = rgba[i * 4 + 0];
        bgra[i * 4 + 3] = rgba[i * 4 + 3];
    }

    file.write(reinterpret_cast<const char *>(bgra.data()), bgra.size());
    return true;
}

static bool saveTextureToPNG(id<MTLTexture> texture, const std::string &outPNGPath) {
    uint32_t w = static_cast<uint32_t>(texture.width);
    uint32_t h = static_cast<uint32_t>(texture.height);

    std::vector<float> floatPixels(w * h * 4);
    [texture getBytes:floatPixels.data()
          bytesPerRow:w * 4 * sizeof(float)
         fromRegion:MTLRegionMake2D(0, 0, w, h)
        mipmapLevel:0];

    std::vector<uint8_t> bytePixels(w * h * 4);
    for (size_t i = 0; i < w * h; i++) {
        float r = std::min(1.0f, std::max(0.0f, floatPixels[i * 4 + 0]));
        float g = std::min(1.0f, std::max(0.0f, floatPixels[i * 4 + 1]));
        float b = std::min(1.0f, std::max(0.0f, floatPixels[i * 4 + 2]));
        bytePixels[i * 4 + 0] = static_cast<uint8_t>(r * 255.0f);
        bytePixels[i * 4 + 1] = static_cast<uint8_t>(g * 255.0f);
        bytePixels[i * 4 + 2] = static_cast<uint8_t>(b * 255.0f);
        bytePixels[i * 4 + 3] = 255;
    }

    std::string tmpTGA = outPNGPath + ".tmp.tga";
    saveTGA(tmpTGA, w, h, bytePixels);
    std::string sipsCmd = "sips -s format png " + tmpTGA + " --out " + outPNGPath + " > /dev/null 2>&1 && rm -f " + tmpTGA;
    int ret = system(sipsCmd.c_str());
    return (ret == 0);
}

struct RendererState {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;

    id<MTLComputePipelineState> cascadeGatherPipeline;
    id<MTLComputePipelineState> cascadeIntegratePipeline;
    id<MTLComputePipelineState> filterCascadePipeline;
    id<MTLComputePipelineState> causticsPipeline;
    id<MTLComputePipelineState> filterPipeline;
    id<MTLComputePipelineState> scenePipeline;
    id<MTLComputePipelineState> pathTracePipeline;

    id<MTLTexture> irradianceAtlas;
    id<MTLTexture> filteredIrradianceAtlas;
    id<MTLTexture> cascadeTex[4];
    id<MTLTexture> dummyCascadeTexture;
    id<MTLBuffer> cascadeParamsBuffer[4];
    id<MTLBuffer> cascadeParamsSingle;   // level 0 stretched over the whole range, for the merge ablation
    id<MTLTexture> causticTexture;
    id<MTLBuffer> causticBuffer;
    size_t causticBufferSize = 0;

    id<MTLBuffer> uniformBuffer;
    id<MTLBuffer> teapotNodeBuffer;
    id<MTLBuffer> teapotTriBuffer;
    uint32_t numTeapotNodes = 0;
    uint32_t numTeapotTris = 0;

    OrbitCamera camera;
    float sunTime = 0.0f;
    float sunSpeed = 0.45f;
    bool sunPaused = true;
    float manualAzimuthOffset = 0.0f;
    float manualElevationOffset = 0.0f;

    uint32_t renderMode = 1;
    float glassRoughness = 0.35f;
    uint32_t frameCount = 0;

    double lastFrameTime = 0.0;
    double fps = 60.0;
    double gpuDurationMs = 0.0;
    bool showStatsOverlay = true;

    uint32_t currentWidth = 1280;
    uint32_t currentHeight = 720;
    // Path-traced reference (mode 4). The accumulation buffer is progressive:
    // it keeps integrating until something invalidates it.
    id<MTLTexture> ptAccumTexture;
    id<MTLBuffer> ptParamsBuffer;
    uint32_t ptSampleCount = 0;
    uint32_t ptSamplesPerLaunch = 1;
    uint32_t ptMaxDepth = 12;
    uint32_t ptRRStartDepth = 5;
    float ptSunAngularRadiusDeg = 0.5f;
    float ptIndirectClamp = 0.0f;
    uint32_t ptSeedOffset = 0;
    uint32_t ptOpticsMode = 1;   // which mode's optics the reference should model
    uint32_t ablationMask = kAblAll;
    // How many internal reflections a refracted ray may make inside a dielectric
    // before the remaining energy is dropped. 1 is the old single-pair behaviour.
    uint32_t glassBounces = 4;

    uint32_t brightnessMode = 2; // 6 modes: 0..5 (0.8x, 1.8x, 2.8x, 4.2x, 6.5x, 10.0x)
    uint32_t lightColorMode = 0; // 3 modes: 0 (Normal), 1 (Smooth RGB), 2 (Stepped RGB)
    float animTime = 0.0f;
};

static RendererState gRenderer;

static std::string resolveExistingPath(const std::vector<std::string> &candidates) {
    for (const auto &c : candidates) {
        std::ifstream f(c.c_str());
        if (f.good()) return c;
    }
    return candidates.empty() ? "" : candidates[0];
}

// Private textures come back with undefined contents, and both the irradiance
// atlas and the caustic target are read before they are first fully written.
static void clearTextures(RendererState &state, NSArray<id<MTLTexture>> *textures) {
    id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
    for (id<MTLTexture> tex in textures) {
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = tex;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        [[cmd renderCommandEncoderWithDescriptor:pass] endEncoding];
    }
    [cmd commit];
    [cmd waitUntilCompleted];
}

bool initMetal(RendererState &state, const std::string &shaderPath, const std::string &teapotBinPath) {
    state.device = MTLCreateSystemDefaultDevice();
    if (!state.device) {
        std::cerr << "[Metal] Failed to find Metal device.\n";
        return false;
    }
    std::cout << "[Metal] GPU Device: " << [[state.device name] UTF8String] << "\n";

    state.commandQueue = [state.device newCommandQueue];

    NSError *error = nil;
    id<MTLLibrary> library = nil;

    std::string resolvedMetallib = resolveExistingPath({
        "Metal/shaders/RCGlassShaders.metallib",
        "shaders/RCGlassShaders.metallib",
        "RCGlassShaders.metallib"
    });

    if (!resolvedMetallib.empty()) {
        NSString *metallibPath = [NSString stringWithUTF8String:resolvedMetallib.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
            NSURL *libURL = [NSURL fileURLWithPath:metallibPath];
            library = [state.device newLibraryWithURL:libURL error:&error];
        }
    }

    if (!library) {
        NSString *srcPath = [NSString stringWithUTF8String:shaderPath.c_str()];
        NSString *shaderSource = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&error];
        if (!shaderSource) {
            std::cerr << "[Metal] Error reading shader source at " << shaderPath << ": " << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        library = [state.device newLibraryWithSource:shaderSource options:opts error:&error];
        if (!library) {
            std::cerr << "[Metal] Shader compilation error: " << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }
    }

    id<MTLFunction> cascadeGatherFunc    = [library newFunctionWithName:@"cascadeGatherKernel"];
    id<MTLFunction> cascadeIntegrateFunc = [library newFunctionWithName:@"cascadeIntegrateKernel"];
    id<MTLFunction> filterCascadeFunc    = [library newFunctionWithName:@"filterIrradianceAtlasKernel"];
    id<MTLFunction> causticsFunc      = [library newFunctionWithName:@"generateCausticsKernel"];
    id<MTLFunction> filterFunc        = [library newFunctionWithName:@"filterCausticsKernel"];
    id<MTLFunction> sceneFunc         = [library newFunctionWithName:@"renderSceneKernel"];
    id<MTLFunction> pathTraceFunc     = [library newFunctionWithName:@"pathTraceKernel"];

    if (!cascadeGatherFunc || !cascadeIntegrateFunc || !filterCascadeFunc || !causticsFunc || !filterFunc || !sceneFunc || !pathTraceFunc) {
        std::cerr << "[Metal] Failed to locate required kernel functions in library.\n";
        return false;
    }

    auto makePipeline = [&](id<MTLFunction> fn, const char *label) -> id<MTLComputePipelineState> {
        NSError *err = nil;
        id<MTLComputePipelineState> pso = [state.device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            std::cerr << "[Metal] Pipeline '" << label << "' failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "unknown error") << "\n";
        }
        return pso;
    };

    state.cascadeGatherPipeline    = makePipeline(cascadeGatherFunc, "cascadeGather");
    state.cascadeIntegratePipeline = makePipeline(cascadeIntegrateFunc, "cascadeIntegrate");
    state.filterCascadePipeline    = makePipeline(filterCascadeFunc, "filterIrradiance");
    state.causticsPipeline         = makePipeline(causticsFunc, "generateCaustics");
    state.filterPipeline           = makePipeline(filterFunc, "filterCaustics");
    state.scenePipeline            = makePipeline(sceneFunc, "renderScene");
    state.pathTracePipeline        = makePipeline(pathTraceFunc, "pathTrace");

    if (!state.cascadeGatherPipeline || !state.cascadeIntegratePipeline || !state.filterCascadePipeline ||
        !state.causticsPipeline || !state.filterPipeline || !state.scenePipeline || !state.pathTracePipeline) {
        return false;
    }

    MTLTextureDescriptor *atlasDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                         width:320
                                                                                        height:64
                                                                                     mipmapped:NO];
    atlasDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    atlasDesc.storageMode = MTLStorageModePrivate;
    state.irradianceAtlas = [state.device newTextureWithDescriptor:atlasDesc];
    state.filteredIrradianceAtlas = [state.device newTextureWithDescriptor:atlasDesc];

    // One array texture per cascade level (5 slices, one per room surface),
    // sized probesPerAxis x probesPerAxis probes with `rays` directions packed
    // per probe along X. A single (non-array) texture would exceed Metal's
    // 16384-wide limit at the coarser levels once the ray count grows large.
    // Each is fully overwritten by cascadeGatherKernel every frame, so unlike
    // the atlas textures above they don't need a startup clear.
    for (int level = 0; level < 4; level++) {
        const CascadeLevelSpec &spec = kCascadeLevelSpecs[level];
        uint32_t w = spec.probesPerAxis * spec.rays;
        uint32_t h = spec.probesPerAxis;
        MTLTextureDescriptor *cascDesc = [[MTLTextureDescriptor alloc] init];
        cascDesc.textureType = MTLTextureType2DArray;
        cascDesc.pixelFormat = MTLPixelFormatRGBA32Float;
        cascDesc.width = w;
        cascDesc.height = h;
        cascDesc.arrayLength = 5;
        cascDesc.mipmapLevelCount = 1;
        cascDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        cascDesc.storageMode = MTLStorageModePrivate;
        state.cascadeTex[level] = [state.device newTextureWithDescriptor:cascDesc];

        CascadeLevelParams params;
        params.tMin = kCascadeRanges[level];
        params.tMax = kCascadeRanges[level + 1];
        params.raysThisLevel = spec.rays;
        params.probesPerAxisThisLevel = spec.probesPerAxis;
        if (level == 3) {
            params.raysUpperLevel = 0;
            params.probesPerAxisUpperLevel = 0;
            params.skyBoost = 1.0f;
        } else {
            params.raysUpperLevel = kCascadeLevelSpecs[level + 1].rays;
            params.probesPerAxisUpperLevel = kCascadeLevelSpecs[level + 1].probesPerAxis;
            params.skyBoost = 0.0f;
        }
        state.cascadeParamsBuffer[level] = [state.device newBufferWithBytes:&params
                                                                      length:sizeof(CascadeLevelParams)
                                                                     options:MTLResourceStorageModeShared];
    }

    // Cascade ablation: level 0's probe grid and ray count, but covering the
    // whole [0.005, 100] range in one gather with nothing above it to merge.
    {
        CascadeLevelParams single;
        single.tMin = kCascadeRanges[0];
        single.tMax = kCascadeRanges[4];
        single.raysThisLevel = kCascadeLevelSpecs[0].rays;
        single.raysUpperLevel = 0;
        single.probesPerAxisThisLevel = kCascadeLevelSpecs[0].probesPerAxis;
        single.probesPerAxisUpperLevel = 0;
        single.skyBoost = 1.0f;
        state.cascadeParamsSingle = [state.device newBufferWithBytes:&single
                                                              length:sizeof(CascadeLevelParams)
                                                             options:MTLResourceStorageModeShared];
    }

    MTLTextureDescriptor *dummyDesc = [[MTLTextureDescriptor alloc] init];
    dummyDesc.textureType = MTLTextureType2DArray;
    dummyDesc.pixelFormat = MTLPixelFormatRGBA32Float;
    dummyDesc.width = 1;
    dummyDesc.height = 1;
    dummyDesc.arrayLength = 1;
    dummyDesc.mipmapLevelCount = 1;
    dummyDesc.usage = MTLTextureUsageShaderRead;
    dummyDesc.storageMode = MTLStorageModePrivate;
    state.dummyCascadeTexture = [state.device newTextureWithDescriptor:dummyDesc];

    const uint32_t causticRes = 1024;
    MTLTextureDescriptor *cTexDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                        width:causticRes
                                                                                       height:causticRes
                                                                                    mipmapped:NO];
    cTexDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    cTexDesc.storageMode = MTLStorageModePrivate;
    state.causticTexture = [state.device newTextureWithDescriptor:cTexDesc];

    state.causticBufferSize = causticRes * causticRes * 4 * sizeof(uint32_t);
    state.causticBuffer = [state.device newBufferWithLength:state.causticBufferSize options:MTLResourceStorageModePrivate];

    clearTextures(state, @[ state.irradianceAtlas, state.filteredIrradianceAtlas, state.causticTexture ]);

    state.uniformBuffer = [state.device newBufferWithLength:sizeof(GlassUniforms)
                                                    options:MTLResourceStorageModeShared];
    state.ptParamsBuffer = [state.device newBufferWithLength:sizeof(PathTraceParams)
                                                     options:MTLResourceStorageModeShared];

    TeapotMesh teapot;
    if (teapot.loadFromBinary(teapotBinPath)) {
        state.numTeapotNodes = static_cast<uint32_t>(teapot.nodes.size());
        state.numTeapotTris  = static_cast<uint32_t>(teapot.triangles.size());

        state.teapotNodeBuffer = [state.device newBufferWithBytes:teapot.nodes.data()
                                                           length:teapot.nodes.size() * sizeof(GPUBVHNode)
                                                          options:MTLResourceStorageModeShared];

        state.teapotTriBuffer  = [state.device newBufferWithBytes:teapot.triangles.data()
                                                           length:teapot.triangles.size() * sizeof(GPUTriangle)
                                                          options:MTLResourceStorageModeShared];
    } else {
        std::cerr << "[Metal] Warning: Teapot geometry not loaded; running with analytic objects.\n";
    }

    return true;
}

static inline simd::float3 hsvToRgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r = 0, g = 0, b = 0;
    int i = static_cast<int>(h * 6.0f) % 6;
    switch (i) {
        case 0: r = c; g = x; b = 0; break;
        case 1: r = x; g = c; b = 0; break;
        case 2: r = 0; g = c; b = x; break;
        case 3: r = 0; g = x; b = c; break;
        case 4: r = x; g = 0; b = c; break;
        case 5: r = c; g = 0; b = x; break;
    }
    return simd::make_float3(r + m, g + m, b + m);
}

static void resetPathTraceAccumulation(RendererState &state) {
    state.ptSampleCount = 0;
}

// The reference accumulates in its own full-precision buffer, so it has to
// follow the target's size rather than the drawable's.
static void ensurePathTraceTarget(RendererState &state, uint32_t width, uint32_t height) {
    if (state.ptAccumTexture &&
        state.ptAccumTexture.width == width &&
        state.ptAccumTexture.height == height) {
        return;
    }
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
#if TARGET_OS_OSX
    // Managed rather than private: the offline comparison reads the raw HDR
    // accumulation back to compute metrics before tone mapping.
    desc.storageMode = MTLStorageModeManaged;
#endif
    state.ptAccumTexture = [state.device newTextureWithDescriptor:desc];
    resetPathTraceAccumulation(state);
}

void renderFrame(
    RendererState &state,
    id<MTLTexture> targetTexture,
    id<MTLCommandBuffer> cmdBuffer,
    float deltaTime
) {
    uint32_t width  = static_cast<uint32_t>(targetTexture.width);
    uint32_t height = static_cast<uint32_t>(targetTexture.height);

    state.currentWidth = width;
    state.currentHeight = height;
    state.animTime += deltaTime;
    state.camera.aspect = float(width) / float(height);

    if (!state.sunPaused) {
        state.sunTime += deltaTime * state.sunSpeed;
    }

    float sunAzimuth = state.sunTime * 0.35f + state.manualAzimuthOffset;
    float sunX = -0.78f + std::sin(sunAzimuth) * 0.12f;
    float sunY =  0.55f + std::cos(sunAzimuth * 0.7f) * 0.10f + state.manualElevationOffset;
    float sunZ =  std::cos(sunAzimuth) * 0.42f + 0.05f;
    simd::float3 sunDir = simd::normalize(simd::make_float3(sunX, sunY, sunZ));

    GlassUniforms uniforms;
    simd::float4x4 viewMat = state.camera.getViewMatrix();
    simd::float4x4 projMat = state.camera.getProjectionMatrix();

    uniforms.viewInverse = simd_inverse(viewMat);
    uniforms.projectionInverse = simd_inverse(projMat);
    uniforms.cameraPosition = state.camera.getPosition();
    uniforms.time = state.sunTime;

    uniforms.sunDirection = sunDir;

    // 6 brightness levels on 'B': 0.8x, 1.8x, 2.8x (default), 4.2x, 6.5x, 10.0x
    static const float kBrightnessLevels[6] = { 0.8f, 1.8f, 2.8f, 4.2f, 6.5f, 10.0f };
    uniforms.sunIntensity = kBrightnessLevels[state.brightnessMode % 6];

    // 3 color modes on 'C': Normal, Smooth RGB, Stepped RGB
    if (state.lightColorMode == 1) {
        // Smooth RGB continuous rainbow cycle
        float hue = std::fmod(state.animTime * 0.15f, 1.0f);
        uniforms.sunColor = hsvToRgb(hue, 0.90f, 1.0f);
    } else if (state.lightColorMode == 2) {
        // Stepped sharp RGB switch
        static const simd::float3 kStepColors[6] = {
            simd::make_float3(1.0f, 0.12f, 0.12f),  // Red
            simd::make_float3(0.12f, 1.0f, 0.12f),  // Green
            simd::make_float3(0.15f, 0.45f, 1.0f),  // Blue
            simd::make_float3(1.0f, 0.92f, 0.12f),  // Yellow
            simd::make_float3(0.12f, 1.0f, 0.95f),  // Cyan
            simd::make_float3(1.0f, 0.15f, 0.95f)   // Magenta
        };
        int stepIdx = static_cast<int>(state.animTime / 0.85f) % 6;
        uniforms.sunColor = kStepColors[stepIdx];
    } else {
        // Normal warm sunlight
        uniforms.sunColor = simd::make_float3(1.0f, 0.98f, 0.92f);
    }
    uniforms.ambientIntensity = 0.25f;

    uniforms.glassIor = 1.52f;
    uniforms.glassDispersion = 0.025f;
    uniforms.glassRoughness = state.glassRoughness;
    uniforms.glassAbsorption = 0.05f;

    uniforms.renderMode = state.renderMode;
    uniforms.width = width;
    uniforms.height = height;
    uniforms.frameIndex = state.frameCount++;

    uniforms.numTeapotNodes = state.numTeapotNodes;
    uniforms.numTeapotTris  = state.numTeapotTris;
    uniforms.ablationMask = state.ablationMask;
    uniforms.glassBounces = state.glassBounces;

    id<MTLBuffer> uniformBuffer = state.uniformBuffer;
    memcpy(uniformBuffer.contents, &uniforms, sizeof(GlassUniforms));

    // Mode 4 is the reference: no cascades, no splatting, no atlas - just paths.
    if (state.renderMode == 4) {
        ensurePathTraceTarget(state, width, height);

        // The reference reuses the raster path's optical parameters: which
        // dispersion strength, and whether the interface is rough. Mode 0 is
        // single-IOR, so its ablation bit for dispersion is cleared here.
        uniforms.renderMode = state.ptOpticsMode;
        if (state.ptOpticsMode == 0) uniforms.ablationMask &= ~uint32_t(kAblDispersion);
        memcpy(uniformBuffer.contents, &uniforms, sizeof(GlassUniforms));

        PathTraceParams ptParams;
        ptParams.samplesPerLaunch = std::max(1u, state.ptSamplesPerLaunch);
        ptParams.sampleBase = state.ptSampleCount;
        ptParams.maxDepth = state.ptMaxDepth;
        ptParams.rrStartDepth = state.ptRRStartDepth;
        ptParams.sunAngularRadius = state.ptSunAngularRadiusDeg * float(M_PI) / 180.0f;
        ptParams.indirectClamp = state.ptIndirectClamp;
        ptParams.exposure = 1.0f;
        ptParams.seedOffset = state.ptSeedOffset;
        memcpy(state.ptParamsBuffer.contents, &ptParams, sizeof(PathTraceParams));

        id<MTLComputeCommandEncoder> ptEnc = [cmdBuffer computeCommandEncoder];
        [ptEnc setComputePipelineState:state.pathTracePipeline];
        [ptEnc setTexture:state.ptAccumTexture atIndex:0];
        [ptEnc setTexture:targetTexture atIndex:1];
        [ptEnc setBuffer:uniformBuffer offset:0 atIndex:0];
        [ptEnc setBuffer:state.ptParamsBuffer offset:0 atIndex:1];
        if (state.teapotNodeBuffer) [ptEnc setBuffer:state.teapotNodeBuffer offset:0 atIndex:2];
        if (state.teapotTriBuffer)  [ptEnc setBuffer:state.teapotTriBuffer offset:0 atIndex:3];

        MTLSize ptTg = MTLSizeMake(8, 8, 1);
        MTLSize ptGrid = MTLSizeMake((width + 7) / 8, (height + 7) / 8, 1);
        [ptEnc dispatchThreadgroups:ptGrid threadsPerThreadgroup:ptTg];
        [ptEnc endEncoding];

        state.ptSampleCount += ptParams.samplesPerLaunch;
        return;
    }

    MTLSize rcTg = MTLSizeMake(16, 16, 1);
    MTLSize rcGrid = MTLSizeMake((320 + 15) / 16, (64 + 15) / 16, 1);

    const bool useCascadeGI    = (state.ablationMask & kAblCascadeGI) != 0;
    const bool mergeCascades   = (state.ablationMask & kAblCascadeMerge) != 0;
    const bool filterAtlas     = (state.ablationMask & kAblAtlasFilter) != 0;
    const bool splatCaustics   = (state.ablationMask & kAblCaustics) != 0;

    if (state.renderMode != 0 && useCascadeGI && state.cascadeGatherPipeline && state.irradianceAtlas) {
        // Far-to-near: level 3 has nothing above it to read, level 0 reads level 1,
        // and so on down. Each dispatch covers exactly that level's probe x ray grid.
        // With the merge ablated away only level 0 runs, stretched over the full
        // range, which is the flat single-gather this hierarchy replaced.
        for (int level = mergeCascades ? 3 : 0; level >= 0; level--) {
            const CascadeLevelSpec &spec = kCascadeLevelSpecs[level];
            id<MTLTexture> upperTex = (level == 3 || !mergeCascades) ? state.dummyCascadeTexture : state.cascadeTex[level + 1];
            id<MTLBuffer> levelParams = mergeCascades ? state.cascadeParamsBuffer[level] : state.cascadeParamsSingle;

            id<MTLComputeCommandEncoder> cgEnc = [cmdBuffer computeCommandEncoder];
            [cgEnc setComputePipelineState:state.cascadeGatherPipeline];
            [cgEnc setTexture:state.cascadeTex[level] atIndex:0];
            [cgEnc setTexture:upperTex atIndex:1];
            [cgEnc setTexture:(filterAtlas ? state.filteredIrradianceAtlas : state.irradianceAtlas) atIndex:2];
            [cgEnc setBuffer:uniformBuffer offset:0 atIndex:0];
            [cgEnc setBuffer:levelParams offset:0 atIndex:1];
            if (state.teapotNodeBuffer) [cgEnc setBuffer:state.teapotNodeBuffer offset:0 atIndex:2];
            if (state.teapotTriBuffer)  [cgEnc setBuffer:state.teapotTriBuffer offset:0 atIndex:3];

            uint32_t levelWidth = spec.probesPerAxis * spec.rays;
            MTLSize cgGrid = MTLSizeMake((levelWidth + 15) / 16, (spec.probesPerAxis + 15) / 16, 5);
            [cgEnc dispatchThreadgroups:cgGrid threadsPerThreadgroup:rcTg];
            [cgEnc endEncoding];
        }

        if (state.cascadeIntegratePipeline) {
            id<MTLComputeCommandEncoder> ciEnc = [cmdBuffer computeCommandEncoder];
            [ciEnc setComputePipelineState:state.cascadeIntegratePipeline];
            [ciEnc setTexture:state.irradianceAtlas atIndex:0];
            [ciEnc setTexture:state.cascadeTex[0] atIndex:1];
            [ciEnc setBuffer:uniformBuffer offset:0 atIndex:0];
            [ciEnc setBuffer:state.cascadeParamsBuffer[0] offset:0 atIndex:1];
            [ciEnc dispatchThreadgroups:rcGrid threadsPerThreadgroup:rcTg];
            [ciEnc endEncoding];
        }

        if (filterAtlas && state.filterCascadePipeline && state.filteredIrradianceAtlas) {
            id<MTLComputeCommandEncoder> fcEnc = [cmdBuffer computeCommandEncoder];
            [fcEnc setComputePipelineState:state.filterCascadePipeline];
            [fcEnc setTexture:state.irradianceAtlas atIndex:0];
            [fcEnc setTexture:state.filteredIrradianceAtlas atIndex:1];
            [fcEnc dispatchThreadgroups:rcGrid threadsPerThreadgroup:rcTg];
            [fcEnc endEncoding];
        }
    }

    id<MTLBlitCommandEncoder> clearBlit = [cmdBuffer blitCommandEncoder];
    [clearBlit fillBuffer:state.causticBuffer range:NSMakeRange(0, state.causticBufferSize) value:0];
    [clearBlit endEncoding];

    if (state.renderMode != 0 && splatCaustics) {
        id<MTLComputeCommandEncoder> cEnc = [cmdBuffer computeCommandEncoder];
        [cEnc setComputePipelineState:state.causticsPipeline];
        [cEnc setBuffer:state.causticBuffer offset:0 atIndex:0];
        [cEnc setBuffer:uniformBuffer offset:0 atIndex:1];
        if (state.teapotNodeBuffer) [cEnc setBuffer:state.teapotNodeBuffer offset:0 atIndex:2];
        if (state.teapotTriBuffer)  [cEnc setBuffer:state.teapotTriBuffer offset:0 atIndex:3];

        MTLSize cTg = MTLSizeMake(16, 16, 1);
        MTLSize cGrid = MTLSizeMake((2048 + 15) / 16, (2048 + 15) / 16, 1);
        [cEnc dispatchThreadgroups:cGrid threadsPerThreadgroup:cTg];
        [cEnc endEncoding];
    }

    if (splatCaustics) {
        id<MTLComputeCommandEncoder> fEnc = [cmdBuffer computeCommandEncoder];
        [fEnc setComputePipelineState:state.filterPipeline];
        [fEnc setBuffer:state.causticBuffer offset:0 atIndex:0];
        [fEnc setTexture:state.causticTexture atIndex:0];
        [fEnc setBuffer:uniformBuffer offset:0 atIndex:1];

        MTLSize fTg = MTLSizeMake(16, 16, 1);
        MTLSize fGrid = MTLSizeMake((1024 + 15) / 16, (1024 + 15) / 16, 1);
        [fEnc dispatchThreadgroups:fGrid threadsPerThreadgroup:fTg];
        [fEnc endEncoding];
    }

    id<MTLComputeCommandEncoder> rEnc = [cmdBuffer computeCommandEncoder];
    [rEnc setComputePipelineState:state.scenePipeline];
    [rEnc setTexture:targetTexture atIndex:0];
    [rEnc setTexture:state.causticTexture atIndex:1];
    id<MTLTexture> atlasToSample = (filterAtlas && state.filteredIrradianceAtlas)
                                 ? state.filteredIrradianceAtlas
                                 : state.irradianceAtlas;
    if (atlasToSample) {
        [rEnc setTexture:atlasToSample atIndex:2];
    } else {
        [rEnc setTexture:state.causticTexture atIndex:2];
    }
    [rEnc setBuffer:uniformBuffer offset:0 atIndex:0];
    if (state.teapotNodeBuffer) [rEnc setBuffer:state.teapotNodeBuffer offset:0 atIndex:1];
    if (state.teapotTriBuffer)  [rEnc setBuffer:state.teapotTriBuffer offset:0 atIndex:2];

    MTLSize rTg = MTLSizeMake(16, 16, 1);
    MTLSize rGrid = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
    [rEnc dispatchThreadgroups:rGrid threadsPerThreadgroup:rTg];
    [rEnc endEncoding];
}

@interface RCWindow : NSWindow
@end

@implementation RCWindow
- (BOOL)canBecomeKeyWindow { return YES; }
- (BOOL)canBecomeMainWindow { return YES; }
@end

struct SystemMetrics {
    double procRamMB = 0.0;
    double sysUsedRamGB = 0.0;
    double sysTotalRamGB = 0.0;
    double appVramMB = 0.0;
    double totalVramMB = 0.0;
    int ioGpuLoad = -1;
};

static inline NSString *getGpuTypeDescription(id<MTLDevice> device) {
    if (!device) return @"Unknown";
    if (device.hasUnifiedMemory) {
        return @"Apple Silicon (Unified Memory)";
    } else if (device.isLowPower) {
        return @"Integrated GPU";
    } else {
        return @"Discrete GPU (PCIe)";
    }
}

static inline SystemMetrics querySystemMetrics(id<MTLDevice> device) {
    SystemMetrics m;
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        m.procRamMB = info.resident_size / (1024.0 * 1024.0);
    }
    uint64_t totalRamBytes = 0;
    size_t len = sizeof(totalRamBytes);
    sysctlbyname("hw.memsize", &totalRamBytes, &len, NULL, 0);
    m.sysTotalRamGB = totalRamBytes / (1024.0 * 1024.0 * 1024.0);

    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_count) == KERN_SUCCESS) {
        uint64_t usedRamBytes = (vm_stat.active_count + vm_stat.wire_count + vm_stat.speculative_count) * (uint64_t)vm_page_size;
        m.sysUsedRamGB = usedRamBytes / (1024.0 * 1024.0 * 1024.0);
    }
    if (device) {
        m.appVramMB = [device currentAllocatedSize] / (1024.0 * 1024.0);
        m.totalVramMB = [device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0);
    }
    CFMutableDictionaryRef matching = IOServiceMatching("IOAccelerator");
    io_iterator_t iterator;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS) {
        io_registry_entry_t entry;
        while ((entry = IOIteratorNext(iterator))) {
            CFMutableDictionaryRef props = nullptr;
            if (IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, kNilOptions) == KERN_SUCCESS && props) {
                NSDictionary *dict = (__bridge NSDictionary *)props;
                NSDictionary *perf = dict[@"PerformanceStatistics"];
                if (perf && perf[@"Device Utilization %"]) {
                    int val = [perf[@"Device Utilization %"] intValue];
                    if (val > m.ioGpuLoad) m.ioGpuLoad = val;
                }
                CFRelease(props);
            }
            IOObjectRelease(entry);
        }
        IOObjectRelease(iterator);
    }
    return m;
}

@interface RCStatsOverlayView : NSView
@property (nonatomic, strong) NSTextField *textField;
- (void)updateWithRenderer:(const RendererState &)state;
@end

@implementation RCStatsOverlayView

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        self.wantsLayer = YES;
        self.layer.backgroundColor = [[NSColor colorWithCalibratedRed:0.06 green:0.08 blue:0.12 alpha:0.86] CGColor];
        self.layer.cornerRadius = 10.0;
        self.layer.borderWidth = 1.0;
        self.layer.borderColor = [[NSColor colorWithCalibratedRed:0.25 green:0.35 blue:0.50 alpha:0.35] CGColor];
        self.layer.shadowColor = [[NSColor blackColor] CGColor];
        self.layer.shadowOpacity = 0.45;
        self.layer.shadowRadius = 8.0;
        self.layer.shadowOffset = CGSizeMake(0, -3);

        NSRect textFrame = NSInsetRect(self.bounds, 14, 10);
        self.textField = [[NSTextField alloc] initWithFrame:textFrame];
        self.textField.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        self.textField.editable = NO;
        self.textField.selectable = NO;
        self.textField.bezeled = NO;
        self.textField.drawsBackground = NO;

        NSFont *font = [NSFont monospacedSystemFontOfSize:11.5 weight:NSFontWeightMedium];
        if (!font) font = [NSFont userFixedPitchFontOfSize:11.5];
        self.textField.font = font;
        self.textField.textColor = [NSColor colorWithCalibratedRed:0.90 green:0.94 blue:0.98 alpha:1.0];
        [self addSubview:self.textField];
    }
    return self;
}

- (NSView *)hitTest:(NSPoint)point {
    return nil;
}

- (void)updateWithRenderer:(const RendererState &)state {
    double frameTimeMs = (state.fps > 0.0) ? (1000.0 / state.fps) : 0.0;
    double gpuDutyCycle = (frameTimeMs > 0.001) ? (state.gpuDurationMs / frameTimeMs) * 100.0 : 0.0;
    gpuDutyCycle = std::min(100.0, std::max(0.0, gpuDutyCycle));

    SystemMetrics m = querySystemMetrics(state.device);
    double effectiveGpuLoad = (m.ioGpuLoad >= 0) ? std::max((double)m.ioGpuLoad, gpuDutyCycle) : gpuDutyCycle;

    const char *modeNames[] = {
        "Mode 0: Whitted RT Baseline",
        "Mode 1: Clear Glass + Caustics",
        "Mode 2: Frosted Rough Glass",
        "Mode 3: High Dispersion Prism",
        "Mode 4: Path Traced Reference"
    };
    const char *modeStr = (state.renderMode <= 4) ? modeNames[state.renderMode] : "Custom";

    static const char *colorModeNames[] = {
        "Normal (Warm Sun)",
        "Smooth RGB Rainbow",
        "Stepped Sharp RGB"
    };
    static const float kBrightnessLevels[6] = { 0.8f, 1.8f, 2.8f, 4.2f, 6.5f, 10.0f };
    const char *colorStr = (state.lightColorMode < 3) ? colorModeNames[state.lightColorMode] : "Normal";
    float curBrightness = kBrightnessLevels[state.brightnessMode % 6];

    NSString *str = [NSString stringWithFormat:
        @"Resolution:  %ux%u\n"
        @"GPU:         %@\n"
        @"Model:       %@\n"
        @"GPU Load:    %3.0f%%  (GPU Time: %4.1f ms)\n"
        @"Framerate:   %4.1f FPS  (%4.1f ms)\n"
        @"VRAM:        %4.0f MB alloc  / %4.0f MB max\n"
        @"RAM:         %4.0f MB app    | %4.1f / %4.1f GB sys\n"
        @"Mode:        %s\n"
        @"Roughness:   %.2f  (glass bounces: %u)\n"
        @"Brightness:  Mode %u/6 (%.1fx)\n"
        @"Light Color: %s%@",
        state.currentWidth, state.currentHeight,
        state.device ? [state.device name] : @"Metal Device",
        getGpuTypeDescription(state.device),
        effectiveGpuLoad, state.gpuDurationMs,
        state.fps, frameTimeMs,
        m.appVramMB, m.totalVramMB,
        m.procRamMB, m.sysUsedRamGB, m.sysTotalRamGB,
        modeStr, state.glassRoughness, state.glassBounces,
        (state.brightnessMode % 6) + 1, curBrightness,
        colorStr,
        (state.renderMode == 4)
            ? [NSString stringWithFormat:@"\nReference:   %u spp accumulated (depth %u, sun %.2f deg)",
                                         state.ptSampleCount, state.ptMaxDepth, state.ptSunAngularRadiusDeg]
            : @""
    ];
    [self.textField setStringValue:str];
}

@end

@interface RCGlassView : MTKView <MTKViewDelegate>
@property (nonatomic, assign) NSPoint lastMousePos;
@property (nonatomic, strong) RCStatsOverlayView *statsOverlay;
- (void)toggleStatsOverlay;
- (void)cycleBrightnessMode;
- (void)cycleLightColorMode;
@end

@implementation RCGlassView

- (instancetype)initWithFrame:(NSRect)frameRect device:(id<MTLDevice>)device {
    self = [super initWithFrame:frameRect device:device];
    if (self) {
        self.delegate = self;
        self.preferredFramesPerSecond = 60;
        self.colorPixelFormat = MTLPixelFormatRGBA16Float;
        self.framebufferOnly = NO;
        gRenderer.lastFrameTime = CACurrentMediaTime();

        NSRect overlayFrame = NSMakeRect(16, frameRect.size.height - 214 - 16, 410, 214);
        self.statsOverlay = [[RCStatsOverlayView alloc] initWithFrame:overlayFrame];
        self.statsOverlay.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
        self.statsOverlay.hidden = !gRenderer.showStatsOverlay;
        [self addSubview:self.statsOverlay];
        [self.statsOverlay updateWithRenderer:gRenderer];
    }
    return self;
}

- (void)toggleStatsOverlay {
    gRenderer.showStatsOverlay = !gRenderer.showStatsOverlay;
    self.statsOverlay.hidden = !gRenderer.showStatsOverlay;
    if (gRenderer.showStatsOverlay) {
        [self.statsOverlay updateWithRenderer:gRenderer];
    }
    std::cout << "[HUD] Statistics overlay: " << (gRenderer.showStatsOverlay ? "SHOWN" : "HIDDEN") << "\n";
}

- (void)cycleBrightnessMode {
    static const float kBrightnessLevels[6] = { 0.8f, 1.8f, 2.8f, 4.2f, 6.5f, 10.0f };
    static const char *kBrightnessNames[6] = {
        "0.8x (Dim Twilight)",
        "1.8x (Soft Light)",
        "2.8x (Normal Default)",
        "4.2x (Bright Sun)",
        "6.5x (Intense Caustics)",
        "10.0x (Overdrive Caustics)"
    };
    gRenderer.brightnessMode = (gRenderer.brightnessMode + 1) % 6;
    resetPathTraceAccumulation(gRenderer);
    std::cout << "[Light] Brightness mode " << (gRenderer.brightnessMode + 1) << "/6: "
              << kBrightnessNames[gRenderer.brightnessMode] << " (" << kBrightnessLevels[gRenderer.brightnessMode] << "x)\n";
    if (gRenderer.showStatsOverlay) {
        [self.statsOverlay updateWithRenderer:gRenderer];
    }
}

- (void)cycleLightColorMode {
    static const char *kColorNames[3] = {
        "Normal (Warm Sunlight)",
        "Smooth RGB Rainbow Cycle",
        "Stepped Sharp RGB Switch"
    };
    gRenderer.lightColorMode = (gRenderer.lightColorMode + 1) % 3;
    resetPathTraceAccumulation(gRenderer);
    std::cout << "[Light] Color mode " << (gRenderer.lightColorMode + 1) << "/3: "
              << kColorNames[gRenderer.lightColorMode] << "\n";
    if (gRenderer.showStatsOverlay) {
        [self.statsOverlay updateWithRenderer:gRenderer];
    }
}

- (BOOL)acceptsFirstResponder { return YES; }

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
}

- (void)drawInMTKView:(MTKView *)view {
    @autoreleasepool {
        double now = CACurrentMediaTime();
        double dt = now - gRenderer.lastFrameTime;
        gRenderer.lastFrameTime = now;
        if (dt > 0.0) {
            double curFps = 1.0 / dt;
            gRenderer.fps = gRenderer.fps * 0.9 + curFps * 0.1;
        }

        id<CAMetalDrawable> drawable = view.currentDrawable;
        if (!drawable) return;

        id<MTLCommandBuffer> cmdBuffer = [gRenderer.commandQueue commandBuffer];
        renderFrame(gRenderer, drawable.texture, cmdBuffer, static_cast<float>(dt));
        [cmdBuffer presentDrawable:drawable];

        [cmdBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            CFTimeInterval gpuStart = cb.GPUStartTime;
            CFTimeInterval gpuEnd = cb.GPUEndTime;
            if (gpuEnd > gpuStart) {
                double durMs = (gpuEnd - gpuStart) * 1000.0;
                gRenderer.gpuDurationMs = (gRenderer.gpuDurationMs == 0.0)
                    ? durMs
                    : (gRenderer.gpuDurationMs * 0.9 + durMs * 0.1);
            }
        }];

        [cmdBuffer commit];

        if (gRenderer.frameCount % 6 == 0 && gRenderer.showStatsOverlay) {
            [self.statsOverlay updateWithRenderer:gRenderer];
        }

        if (gRenderer.frameCount % 60 == 0) {
            NSString *title = [NSString stringWithFormat:@"Radiance Cascades Glass | Mode: %u | FPS: %.1f | Res: %ux%u",
                               gRenderer.renderMode, gRenderer.fps,
                               (uint32_t)view.drawableSize.width, (uint32_t)view.drawableSize.height];
            [self.window setTitle:title];
        }
    }
}

- (void)mouseDown:(NSEvent *)event {
    self.lastMousePos = [event locationInWindow];
}

- (void)rightMouseDown:(NSEvent *)event {
    self.lastMousePos = [event locationInWindow];
}

- (void)mouseDragged:(NSEvent *)event {
    NSPoint currentPos = [event locationInWindow];
    float dx = currentPos.x - self.lastMousePos.x;
    float dy = currentPos.y - self.lastMousePos.y;
    self.lastMousePos = currentPos;

    if (event.modifierFlags & NSEventModifierFlagOption) {
        gRenderer.camera.pan(dx, dy);
    } else {
        gRenderer.camera.orbit(dx, dy);
    }
    resetPathTraceAccumulation(gRenderer);
}

- (void)rightMouseDragged:(NSEvent *)event {
    NSPoint currentPos = [event locationInWindow];
    float dy = currentPos.y - self.lastMousePos.y;
    self.lastMousePos = currentPos;
    gRenderer.camera.zoom(dy * 0.03f);
    resetPathTraceAccumulation(gRenderer);
}

- (void)scrollWheel:(NSEvent *)event {
    float delta = [event scrollingDeltaY];
    gRenderer.camera.zoom(delta * 0.05f);
    resetPathTraceAccumulation(gRenderer);
}

- (void)keyDown:(NSEvent *)event {
    if (event.keyCode == 31) { // 'O' physical keycode across any keyboard layout
        [self toggleStatsOverlay];
        return;
    }
    if (event.keyCode == 11) { // 'B' physical keycode across any keyboard layout
        [self cycleBrightnessMode];
        return;
    }
    if (event.keyCode == 8) { // 'C' physical keycode across any keyboard layout
        [self cycleLightColorMode];
        return;
    }

    NSString *chars = [event charactersIgnoringModifiers];
    if ([chars length] == 0) return;
    unichar c = [chars characterAtIndex:0];

    switch (c) {
        case 'o':
        case 'O':
        case 0x043E: // Ukrainian Cyrillic 'о'
        case 0x041E: // Ukrainian Cyrillic 'О'
            [self toggleStatsOverlay];
            break;
        case 'b':
        case 'B':
        case 0x0431: // Ukrainian 'б'
        case 0x0411: // Ukrainian 'Б'
        case 0x0438: // Ukrainian 'и' (key B on standard keyboard)
        case 0x0418: // Ukrainian 'И'
            [self cycleBrightnessMode];
            break;
        case 'c':
        case 'C':
        case 0x0441: // Ukrainian 'с'
        case 0x0421: // Ukrainian 'С'
            [self cycleLightColorMode];
            break;
        case ' ':
            resetPathTraceAccumulation(gRenderer);
            gRenderer.sunPaused = !gRenderer.sunPaused;
            std::cout << "[Sun] Dynamic motion: " << (gRenderer.sunPaused ? "PAUSED" : "RESUMED") << "\n";
            break;
        case '1':
            gRenderer.renderMode = 1;
            std::cout << "[Mode] Mode 1: Clear Glass + Radiance Cascades\n";
            break;
        case '2':
            gRenderer.renderMode = 2;
            std::cout << "[Mode] Mode 2: Frosted Rough Glass\n";
            break;
        case '3':
            gRenderer.renderMode = 3;
            std::cout << "[Mode] Mode 3: High Spectral Dispersion\n";
            break;
        case '4':
            gRenderer.renderMode = 4;
            resetPathTraceAccumulation(gRenderer);
            std::cout << "[Mode] Mode 4: Path Traced Reference (progressive, "
                      << gRenderer.ptSamplesPerLaunch << " spp/frame, depth "
                      << gRenderer.ptMaxDepth << ")\n";
            break;
        case '0':
            gRenderer.renderMode = 0;
            std::cout << "[Mode] Mode 0: Whitted RT Baseline\n";
            break;
        case '[':
            gRenderer.glassBounces = std::max(1u, gRenderer.glassBounces - 1);
            std::cout << "[Glass] Internal reflection budget: " << gRenderer.glassBounces << "\n";
            break;
        case ']':
            gRenderer.glassBounces = std::min(8u, gRenderer.glassBounces + 1);
            std::cout << "[Glass] Internal reflection budget: " << gRenderer.glassBounces << "\n";
            break;
        case 'r':
        case 'R':
            gRenderer.camera = OrbitCamera();
            resetPathTraceAccumulation(gRenderer);
            std::cout << "[Camera] Reset view\n";
            break;
        case '+':
        case '=':
            gRenderer.glassRoughness = std::min(1.0f, gRenderer.glassRoughness + 0.05f);
            std::cout << "[Roughness] " << gRenderer.glassRoughness << "\n";
            break;
        case '-':
        case '_':
            gRenderer.glassRoughness = std::max(0.0f, gRenderer.glassRoughness - 0.05f);
            std::cout << "[Roughness] " << gRenderer.glassRoughness << "\n";
            break;
        case 's':
        case 'S': {
            uint32_t w = 1920;
            uint32_t h = 1080;
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                            width:w
                                                                                           height:h
                                                                                        mipmapped:NO];
            desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
#if TARGET_OS_OSX
            desc.storageMode = MTLStorageModeManaged;
#endif
            id<MTLTexture> snapTex = [gRenderer.device newTextureWithDescriptor:desc];
            id<MTLCommandBuffer> snapCmd = [gRenderer.commandQueue commandBuffer];
            renderFrame(gRenderer, snapTex, snapCmd, 0.016f);
            if (snapTex.storageMode == MTLStorageModeManaged) {
                id<MTLBlitCommandEncoder> syncBlit = [snapCmd blitCommandEncoder];
                [syncBlit synchronizeResource:snapTex];
                [syncBlit endEncoding];
            }
            [snapCmd commit];
            [snapCmd waitUntilCompleted];

            system("mkdir -p output");
            std::string outPath = "output/rc_glass_snapshot.png";
            if (saveTextureToPNG(snapTex, outPath)) {
                std::cout << "[Capture] Screenshot saved to " << outPath << "\n";
            }
            break;
        }
        case NSLeftArrowFunctionKey:
            gRenderer.manualAzimuthOffset -= 0.08f;
            resetPathTraceAccumulation(gRenderer);
            break;
        case NSRightArrowFunctionKey:
            gRenderer.manualAzimuthOffset += 0.08f;
            resetPathTraceAccumulation(gRenderer);
            break;
        case NSUpArrowFunctionKey:
            gRenderer.manualElevationOffset += 0.04f;
            resetPathTraceAccumulation(gRenderer);
            break;
        case NSDownArrowFunctionKey:
            gRenderer.manualElevationOffset -= 0.04f;
            resetPathTraceAccumulation(gRenderer);
            break;
        case 27:
            [NSApp terminate:nil];
            break;
        default:
            [super keyDown:event];
            break;
    }
}

@end

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property (nonatomic, strong) RCWindow *window;
@property (nonatomic, strong) RCGlassView *view;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    NSRect frame = NSMakeRect(120, 100, 1280, 720);
    NSUInteger styleMask = NSWindowStyleMaskTitled |
                           NSWindowStyleMaskClosable |
                           NSWindowStyleMaskMiniaturizable |
                           NSWindowStyleMaskResizable;

    self.window = [[RCWindow alloc] initWithContentRect:frame
                                              styleMask:styleMask
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"Radiance Cascades Glass & Caustics"];
    self.window.delegate = self;

    self.view = [[RCGlassView alloc] initWithFrame:frame device:gRenderer.device];
    [self.window setContentView:self.view];
    [self.window makeFirstResponder:self.view];

    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];

    std::cout << "Radiance Cascades Glass & Caustics\n\n";
    std::cout << "  Controls:\n";
    std::cout << "    [Left Mouse Drag]   : Orbit Camera\n";
    std::cout << "    [Option + Drag]     : Pan Camera\n";
    std::cout << "    [Right Drag/Scroll] : Zoom Camera\n";
    std::cout << "    [Space]             : Toggle Sun Animation\n";
    std::cout << "    [Arrow Keys]        : Adjust Sun Position\n";
    std::cout << "    [1]                 : Clear Glass Mode\n";
    std::cout << "    [2]                 : Frosted Glass Mode\n";
    std::cout << "    [3]                 : High Dispersion Prism Mode\n";
    std::cout << "    [0]                 : Whitted Ray Tracing Baseline\n";
    std::cout << "    [4]                 : Path Traced Reference (progressive ground truth)\n";
    std::cout << "    [+/-]               : Adjust Roughness\n";
    std::cout << "    [ [ / ] ]           : Internal reflection budget inside glass (1..8)\n";
    std::cout << "    [B]                 : Cycle Light Brightness (6 Modes: 0.8x -> 10.0x)\n";
    std::cout << "    [C]                 : Cycle Light Color (Normal -> Smooth RGB -> Step RGB)\n";
    std::cout << "    [O]                 : Toggle Hardware & Performance Stats Overlay\n";
    std::cout << "    [R]                 : Reset Camera\n";
    std::cout << "    [S]                 : Screenshot (1080p)\n";
    std::cout << "    [ESC]               : Quit\n\n";
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end

// ---------------------------------------------------------------------------
// Reference comparison tooling
//
// Everything below exists to answer four questions about the real-time modes:
// how far they are from a path traced ground truth, how much path tracing fits
// in the same wall clock, what each individual pass is worth, and where the
// approximation stops being defensible. It runs offline, never in the interactive
// loop.
// ---------------------------------------------------------------------------

struct ImageBuffer {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> rgba;
};

struct ImageMetrics {
    double bias = 0.0;   // mean signed luminance error: positive means too bright
    double rmse = 0.0;
    double psnr = 0.0;
    double relMse = 0.0;
    double mae = 0.0;
    double ssim = 0.0;
};

static ImageBuffer readTextureRGBA(id<MTLTexture> tex) {
    ImageBuffer img;
    img.width = static_cast<uint32_t>(tex.width);
    img.height = static_cast<uint32_t>(tex.height);
    img.rgba.resize(size_t(img.width) * img.height * 4);
    [tex getBytes:img.rgba.data()
      bytesPerRow:img.width * 4 * sizeof(float)
       fromRegion:MTLRegionMake2D(0, 0, img.width, img.height)
      mipmapLevel:0];
    return img;
}

static id<MTLTexture> makeReadbackTexture(RendererState &state, uint32_t w, uint32_t h) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
#if TARGET_OS_OSX
    desc.storageMode = MTLStorageModeManaged;
#endif
    return [state.device newTextureWithDescriptor:desc];
}

static void renderFrames(RendererState &state, id<MTLTexture> target, int frames) {
    for (int f = 0; f < frames; f++) {
        id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
        renderFrame(state, target, cmd, 0.016f);
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// Renders one more frame and pulls the result back to the CPU.
static ImageBuffer captureFrame(RendererState &state, id<MTLTexture> target) {
    id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
    renderFrame(state, target, cmd, 0.016f);
    if (target.storageMode == MTLStorageModeManaged) {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit synchronizeResource:target];
        [blit endEncoding];
    }
    [cmd commit];
    [cmd waitUntilCompleted];
    return readTextureRGBA(target);
}

static double timeFramesMs(RendererState &state, id<MTLTexture> target, int warmup, int frames) {
    renderFrames(state, target, warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    renderFrames(state, target, frames);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / double(frames);
}

static inline simd::float3 toneMapACES_CPU(simd::float3 x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    simd::float3 num = x * (a * x + b);
    simd::float3 den = x * (c * x + d) + e;
    simd::float3 r = num / den;
    return simd::make_float3(std::min(1.0f, std::max(0.0f, r.x)),
                             std::min(1.0f, std::max(0.0f, r.y)),
                             std::min(1.0f, std::max(0.0f, r.z)));
}

static ImageBuffer toneMapImage(const ImageBuffer &hdr) {
    ImageBuffer out;
    out.width = hdr.width;
    out.height = hdr.height;
    out.rgba.resize(hdr.rgba.size());
    for (size_t i = 0; i < size_t(hdr.width) * hdr.height; i++) {
        simd::float3 c = simd::make_float3(hdr.rgba[i * 4 + 0], hdr.rgba[i * 4 + 1], hdr.rgba[i * 4 + 2]);
        simd::float3 t = toneMapACES_CPU(c);
        out.rgba[i * 4 + 0] = t.x;
        out.rgba[i * 4 + 1] = t.y;
        out.rgba[i * 4 + 2] = t.z;
        out.rgba[i * 4 + 3] = 1.0f;
    }
    return out;
}

static std::vector<float> lumaOf(const ImageBuffer &img) {
    std::vector<float> l(size_t(img.width) * img.height);
    for (size_t i = 0; i < l.size(); i++) {
        l[i] = 0.2126f * img.rgba[i * 4 + 0] + 0.7152f * img.rgba[i * 4 + 1] + 0.0722f * img.rgba[i * 4 + 2];
    }
    return l;
}

// Separable Gaussian, used only by the SSIM windows.
static std::vector<float> gaussianBlur(const std::vector<float> &src, uint32_t w, uint32_t h, float sigma) {
    int radius = std::max(1, int(std::ceil(sigma * 3.0f)));
    std::vector<float> kernel(2 * radius + 1);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] = std::exp(-float(i * i) / (2.0f * sigma * sigma));
        sum += kernel[i + radius];
    }
    for (auto &k : kernel) k /= sum;

    std::vector<float> tmp(src.size(), 0.0f);
    std::vector<float> dst(src.size(), 0.0f);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int sx = std::min<int>(int(w) - 1, std::max(0, int(x) + i));
                acc += src[size_t(y) * w + sx] * kernel[i + radius];
            }
            tmp[size_t(y) * w + x] = acc;
        }
    }
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int sy = std::min<int>(int(h) - 1, std::max(0, int(y) + i));
                acc += tmp[size_t(sy) * w + x] * kernel[i + radius];
            }
            dst[size_t(y) * w + x] = acc;
        }
    }
    return dst;
}

// Standard SSIM on luminance, Gaussian window sigma 1.5, C1/C2 for a [0,1] range.
static double ssimLuma(const ImageBuffer &a, const ImageBuffer &b) {
    uint32_t w = a.width, h = a.height;
    std::vector<float> x = lumaOf(a), y = lumaOf(b);
    std::vector<float> xx(x.size()), yy(x.size()), xy(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        xx[i] = x[i] * x[i];
        yy[i] = y[i] * y[i];
        xy[i] = x[i] * y[i];
    }
    std::vector<float> mx = gaussianBlur(x, w, h, 1.5f);
    std::vector<float> my = gaussianBlur(y, w, h, 1.5f);
    std::vector<float> mxx = gaussianBlur(xx, w, h, 1.5f);
    std::vector<float> myy = gaussianBlur(yy, w, h, 1.5f);
    std::vector<float> mxy = gaussianBlur(xy, w, h, 1.5f);

    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;
    double acc = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        double m1 = mx[i], m2 = my[i];
        double v1 = std::max(0.0, double(mxx[i]) - m1 * m1);
        double v2 = std::max(0.0, double(myy[i]) - m2 * m2);
        double cv = double(mxy[i]) - m1 * m2;
        double num = (2.0 * m1 * m2 + C1) * (2.0 * cv + C2);
        double den = (m1 * m1 + m2 * m2 + C1) * (v1 + v2 + C2);
        acc += num / den;
    }
    return acc / double(x.size());
}

// Which part of the image a metric is restricted to. The region of every pixel
// comes from the reference's own primary-hit id, so it is the ground truth's
// segmentation, not the approximation's.
enum class Region : uint8_t { All, Floor, Glass };

static std::vector<uint8_t> regionMaskFromIds(const ImageBuffer &accumWithIds) {
    std::vector<uint8_t> mask(size_t(accumWithIds.width) * accumWithIds.height, 0);
    for (size_t i = 0; i < mask.size(); i++) {
        uint32_t id = uint32_t(accumWithIds.rgba[i * 4 + 3] + 0.5f);
        if (id == 2u) mask[i] = uint8_t(Region::Floor);
        else if (id >= 10u) mask[i] = uint8_t(Region::Glass);
        else mask[i] = 255;   // walls, ceiling, sky: neither region
    }
    return mask;
}

// `test` against `reference`, both tone mapped into [0,1] display space. SSIM is
// always global; a windowed index over a scattered pixel set is not meaningful.
static ImageMetrics compareImages(const ImageBuffer &test, const ImageBuffer &reference,
                                  const std::vector<uint8_t> *mask = nullptr,
                                  Region region = Region::All) {
    ImageMetrics m;
    size_t pixels = size_t(reference.width) * reference.height;
    double se = 0.0, ae = 0.0, rel = 0.0, signedLuma = 0.0;
    static const double kLumaWeights[3] = { 0.2126, 0.7152, 0.0722 };
    size_t counted = 0;
    for (size_t i = 0; i < pixels; i++) {
        if (mask && region != Region::All && (*mask)[i] != uint8_t(region)) continue;
        counted++;
        for (int c = 0; c < 3; c++) {
            double a = test.rgba[i * 4 + c];
            double b = reference.rgba[i * 4 + c];
            double d = a - b;
            se += d * d;
            ae += std::fabs(d);
            rel += (d * d) / (b * b + 0.01);
            signedLuma += kLumaWeights[c] * d;
        }
    }
    if (counted == 0) return m;
    double n = double(counted * 3);
    m.rmse = std::sqrt(se / n);
    m.mae = ae / n;
    m.relMse = rel / n;
    m.psnr = (se > 0.0) ? 10.0 * std::log10(1.0 / (se / n)) : 99.0;
    m.bias = signedLuma / double(counted);
    m.ssim = (region == Region::All) ? ssimLuma(test, reference) : 0.0;
    return m;
}

// Absolute luminance difference through a blue->red ramp, at a fixed scale so
// that heat maps from different runs are directly comparable.
static bool saveErrorHeatmap(const ImageBuffer &test, const ImageBuffer &reference,
                             const std::string &path, float fullScale) {
    uint32_t w = reference.width, h = reference.height;
    std::vector<uint8_t> px(size_t(w) * h * 4, 255);
    for (size_t i = 0; i < size_t(w) * h; i++) {
        double d = 0.0;
        for (int c = 0; c < 3; c++) d += std::fabs(double(test.rgba[i * 4 + c]) - double(reference.rgba[i * 4 + c]));
        float t = std::min(1.0f, float(d / 3.0) / fullScale);
        float r, g, b;
        if (t < 0.25f)      { float u = t / 0.25f;          r = 0.0f;      g = 0.0f;      b = 0.25f + 0.75f * u; }
        else if (t < 0.5f)  { float u = (t - 0.25f) / 0.25f; r = 0.0f;      g = u;         b = 1.0f - u;          }
        else if (t < 0.75f) { float u = (t - 0.5f) / 0.25f;  r = u;         g = 1.0f;      b = 0.0f;              }
        else                { float u = (t - 0.75f) / 0.25f; r = 1.0f;      g = 1.0f - u;  b = 0.0f;              }
        px[i * 4 + 0] = uint8_t(r * 255.0f);
        px[i * 4 + 1] = uint8_t(g * 255.0f);
        px[i * 4 + 2] = uint8_t(b * 255.0f);
        px[i * 4 + 3] = 255;
    }
    std::string tmpTGA = path + ".tmp.tga";
    saveTGA(tmpTGA, w, h, px);
    std::string cmd = "sips -s format png " + tmpTGA + " --out " + path + " > /dev/null 2>&1 && rm -f " + tmpTGA;
    return system(cmd.c_str()) == 0;
}

static bool saveImageBufferToPNG(const ImageBuffer &img, const std::string &path) {
    std::vector<uint8_t> px(size_t(img.width) * img.height * 4, 255);
    for (size_t i = 0; i < size_t(img.width) * img.height; i++) {
        for (int c = 0; c < 3; c++) {
            float v = std::min(1.0f, std::max(0.0f, img.rgba[i * 4 + c]));
            px[i * 4 + c] = uint8_t(v * 255.0f);
        }
    }
    std::string tmpTGA = path + ".tmp.tga";
    saveTGA(tmpTGA, img.width, img.height, px);
    std::string cmd = "sips -s format png " + tmpTGA + " --out " + path + " > /dev/null 2>&1 && rm -f " + tmpTGA;
    return system(cmd.c_str()) == 0;
}

struct ReferenceResult {
    ImageBuffer image;              // tone mapped ground truth
    std::vector<uint8_t> region;    // per-pixel Region, from the reference's primary hit
    double noiseFloorRmse;          // RMSE between the two independent halves, halved
    double seconds;
    uint32_t spp;
};

// The accumulator keeps the primary-hit id in alpha, so only RGB is averaged.
static void normaliseAccumulation(ImageBuffer &accum, uint32_t samples) {
    for (size_t i = 0; i < size_t(accum.width) * accum.height; i++) {
        for (int c = 0; c < 3; c++) accum.rgba[i * 4 + c] /= float(samples);
    }
}

// Two independent half-runs, averaged in HDR. Splitting them is what gives the
// reference an honest error bar: a measured RMSE difference of the mode being
// compared is only meaningful above the reference's own residual noise.
static ReferenceResult renderReference(RendererState &state, id<MTLTexture> target,
                                       uint32_t optMode, uint32_t spp, uint32_t chunk) {
    uint32_t savedMode = state.renderMode;
    uint32_t savedSeed = state.ptSeedOffset;
    uint32_t savedChunk = state.ptSamplesPerLaunch;
    uint32_t savedOptics = state.ptOpticsMode;

    // Transport is always full path tracing; the optical mode only selects the
    // dispersion strength and the surface roughness the reference should model.
    state.renderMode = 4;
    state.ptOpticsMode = optMode;

    uint32_t half = std::max(1u, spp / 2);
    ImageBuffer halves[2];

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int side = 0; side < 2; side++) {
        state.ptSeedOffset = (side == 0) ? 0u : 7919u;
        resetPathTraceAccumulation(state);

        uint32_t done = 0;
        while (done < half) {
            uint32_t batch = std::min(chunk, half - done);
            state.ptSamplesPerLaunch = batch;
            id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
            renderFrame(state, target, cmd, 0.016f);
            [cmd commit];
            [cmd waitUntilCompleted];
            done += batch;
        }

        id<MTLCommandBuffer> sync = [state.commandQueue commandBuffer];
        if (state.ptAccumTexture.storageMode == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [sync blitCommandEncoder];
            [blit synchronizeResource:state.ptAccumTexture];
            [blit endEncoding];
        }
        [sync commit];
        [sync waitUntilCompleted];

        ImageBuffer accum = readTextureRGBA(state.ptAccumTexture);
        normaliseAccumulation(accum, half);
        halves[side] = accum;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    ImageBuffer mean = halves[0];
    for (size_t i = 0; i < mean.rgba.size(); i++) {
        mean.rgba[i] = 0.5f * (halves[0].rgba[i] + halves[1].rgba[i]);
    }

    ReferenceResult out;
    out.image = toneMapImage(mean);
    out.region = regionMaskFromIds(halves[0]);
    out.noiseFloorRmse = 0.5 * compareImages(toneMapImage(halves[0]), toneMapImage(halves[1])).rmse;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.spp = half * 2;

    state.renderMode = savedMode;
    state.ptSeedOffset = savedSeed;
    state.ptSamplesPerLaunch = savedChunk;
    state.ptOpticsMode = savedOptics;
    return out;
}

static const char *opticalModeName(uint32_t mode) {
    switch (mode) {
        case 0: return "whitted baseline";
        case 1: return "clear glass";
        case 2: return "frosted glass";
        case 3: return "high dispersion";
        case 4: return "path traced";
        default: return "custom";
    }
}

struct RegionMetrics {
    ImageMetrics all;
    ImageMetrics floor;
    ImageMetrics glass;
};

static RegionMetrics compareAllRegions(const ImageBuffer &test, const ReferenceResult &ref) {
    RegionMetrics r;
    r.all = compareImages(test, ref.image);
    r.floor = compareImages(test, ref.image, &ref.region, Region::Floor);
    r.glass = compareImages(test, ref.image, &ref.region, Region::Glass);
    return r;
}

static void printMetricsHeader() {
    std::cout << "  case                  frame ms      RMSE   PSNR dB    relMSE      SSIM      bias   RMSE floor  RMSE glass\n";
    std::cout << "  --------------------------------------------------------------------------------------------------------\n";
}

static void printMetricsRow(const std::string &label, double frameMs, const RegionMetrics &m) {
    char line[320];
    snprintf(line, sizeof(line), "  %-20s %8.2f  %8.4f  %8.2f  %8.4f  %8.4f  %+8.4f  %10.4f  %10.4f",
             label.c_str(), frameMs, m.all.rmse, m.all.psnr, m.all.relMse, m.all.ssim, m.all.bias,
             m.floor.rmse, m.glass.rmse);
    std::cout << line << "\n";
}

// 1. How far is each real-time mode from a path traced ground truth?
static int runReferenceComparison(RendererState &state, uint32_t width, uint32_t height,
                                  uint32_t refSpp, uint32_t chunk, const std::vector<uint32_t> &modes) {
    system("mkdir -p output");
    id<MTLTexture> target = makeReadbackTexture(state, width, height);

    std::cout << "\n=== Reference comparison ===\n";
    std::cout << "  " << width << "x" << height << ", reference " << refSpp
              << " spp, max depth " << state.ptMaxDepth
              << ", sun disc radius " << state.ptSunAngularRadiusDeg << " deg\n";
    std::cout << "  Metrics are computed on the tone mapped image, in display space.\n\n";

    for (uint32_t mode : modes) {
        ReferenceResult ref = renderReference(state, target, mode, refSpp, chunk);
        std::string refPath = "output/pt_reference_mode" + std::to_string(mode) + ".png";
        saveImageBufferToPNG(ref.image, refPath);

        state.renderMode = mode;
        state.ablationMask = kAblAll;
        double frameMs = timeFramesMs(state, target, 40, 20);
        ImageBuffer rc = captureFrame(state, target);
        RegionMetrics m = compareAllRegions(rc, ref);

        std::string rcPath = "output/cmp_mode" + std::to_string(mode) + "_realtime.png";
        std::string errPath = "output/cmp_mode" + std::to_string(mode) + "_error.png";
        saveImageBufferToPNG(rc, rcPath);
        saveErrorHeatmap(rc, ref.image, errPath, 0.25f);

        std::cout << "  mode " << mode << " (" << opticalModeName(mode) << ")\n";
        printMetricsHeader();
        printMetricsRow("real-time", frameMs, m);
        char note[256];
        snprintf(note, sizeof(note),
                 "  reference: %u spp in %.1f s (%.0f x slower per frame), residual noise RMSE %.4f",
                 ref.spp, ref.seconds, (ref.seconds * 1000.0) / std::max(1e-6, frameMs), ref.noiseFloorRmse);
        std::cout << note << "\n";
        std::cout << "  wrote " << refPath << ", " << rcPath << ", " << errPath << "\n\n";
    }
    return 0;
}

// 2. What does the path tracer get for the same wall clock, and how much does
//    it need before it is as close to ground truth as the real-time frame is?
static int runEqualTimeComparison(RendererState &state, uint32_t width, uint32_t height,
                                 uint32_t refSpp, uint32_t chunk, uint32_t maxSpp, uint32_t mode) {
    system("mkdir -p output");
    id<MTLTexture> target = makeReadbackTexture(state, width, height);

    std::cout << "\n=== Equal-time comparison ===\n";
    std::cout << "  " << width << "x" << height << ", optics from mode " << mode
              << " (" << opticalModeName(mode) << ")\n";

    ReferenceResult ref = renderReference(state, target, mode, refSpp, chunk);
    saveImageBufferToPNG(ref.image, "output/eq_reference.png");

    state.renderMode = mode;
    state.ablationMask = kAblAll;
    double rcMs = timeFramesMs(state, target, 40, 20);
    ImageBuffer rc = captureFrame(state, target);
    ImageMetrics rcMetrics = compareImages(rc, ref.image);
    ImageMetrics rcFloor = compareImages(rc, ref.image, &ref.region, Region::Floor);
    saveImageBufferToPNG(rc, "output/eq_realtime.png");

    char hdr[256];
    snprintf(hdr, sizeof(hdr), "  real-time frame: %.2f ms, RMSE %.4f (floor %.4f), PSNR %.2f dB, SSIM %.4f",
             rcMs, rcMetrics.rmse, rcFloor.rmse, rcMetrics.psnr, rcMetrics.ssim);
    std::cout << hdr << "\n";
    snprintf(hdr, sizeof(hdr), "  reference: %u spp in %.1f s, residual noise RMSE %.4f\n",
             ref.spp, ref.seconds, ref.noiseFloorRmse);
    std::cout << hdr << "\n";

    // Cost per sample, measured on its own short run. Taking it from the whole
    // sweep instead would let one stalled dispatch late in the sweep contaminate
    // the headline equal-time number.
    state.renderMode = 4;
    state.ptOpticsMode = mode;
    state.ptSeedOffset = 101u;
    state.ptSamplesPerLaunch = 1;
    resetPathTraceAccumulation(state);
    renderFrames(state, target, 2);   // warm up
    double msPerSpp = 0.0;
    {
        const int kTimingSamples = 4;
        auto t0 = std::chrono::high_resolution_clock::now();
        renderFrames(state, target, kTimingSamples);
        auto t1 = std::chrono::high_resolution_clock::now();
        msPerSpp = std::chrono::duration<double, std::milli>(t1 - t0).count() / double(kTimingSamples);
    }

    // Progressive sweep with a seed that does not overlap either reference half.
    state.ptSeedOffset = 4242u;
    state.ptSamplesPerLaunch = 1;
    resetPathTraceAccumulation(state);

    std::cout << "  path tracer convergence (same camera, same frame):\n";
    std::cout << "     spp   modelled ms      RMSE     PSNR dB      SSIM   vs real-time frame\n";
    std::cout << "  ---------------------------------------------------------------------\n";

    double elapsedMs = 0.0;
    uint32_t accumulated = 0;
    uint32_t crossoverSpp = 0;

    for (uint32_t nextSpp = 1; nextSpp <= maxSpp; nextSpp *= 2) {
        while (accumulated < nextSpp) {
            uint32_t batch = std::min(chunk, nextSpp - accumulated);
            state.ptSamplesPerLaunch = batch;
            auto t0 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
            renderFrame(state, target, cmd, 0.016f);
            [cmd commit];
            [cmd waitUntilCompleted];
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsedMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            accumulated += batch;
        }

        id<MTLCommandBuffer> sync = [state.commandQueue commandBuffer];
        if (state.ptAccumTexture.storageMode == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [sync blitCommandEncoder];
            [blit synchronizeResource:state.ptAccumTexture];
            [blit endEncoding];
        }
        [sync commit];
        [sync waitUntilCompleted];

        ImageBuffer accum = readTextureRGBA(state.ptAccumTexture);
        normaliseAccumulation(accum, accumulated);
        ImageBuffer ldr = toneMapImage(accum);
        ImageMetrics m = compareImages(ldr, ref.image);

        char row[256];
        snprintf(row, sizeof(row), "  %6u  %12.1f  %8.4f  %8.2f  %8.4f   %s",
                 accumulated, double(accumulated) * msPerSpp, m.rmse, m.psnr, m.ssim,
                 (m.rmse <= rcMetrics.rmse) ? "path tracer is closer" : "still worse");
        std::cout << row << "\n";

        if (crossoverSpp == 0 && m.rmse <= rcMetrics.rmse) {
            crossoverSpp = accumulated;
            saveImageBufferToPNG(ldr, "output/eq_pt_crossover.png");
        }
    }

    // The image the path tracer would actually have on screen in one frame time.
    uint32_t sppInBudget = std::max(1u, uint32_t(std::llround(rcMs / std::max(1e-9, msPerSpp))));
    state.ptSeedOffset = 909u;
    state.ptSamplesPerLaunch = 1;
    resetPathTraceAccumulation(state);
    double budgetMs = 0.0;
    for (uint32_t i = 0; i < sppInBudget; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
        renderFrame(state, target, cmd, 0.016f);
        [cmd commit];
        [cmd waitUntilCompleted];
        auto t1 = std::chrono::high_resolution_clock::now();
        budgetMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    {
        id<MTLCommandBuffer> sync = [state.commandQueue commandBuffer];
        if (state.ptAccumTexture.storageMode == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [sync blitCommandEncoder];
            [blit synchronizeResource:state.ptAccumTexture];
            [blit endEncoding];
        }
        [sync commit];
        [sync waitUntilCompleted];
    }
    ImageBuffer budgetAccum = readTextureRGBA(state.ptAccumTexture);
    normaliseAccumulation(budgetAccum, sppInBudget);
    ImageBuffer budgetImg = toneMapImage(budgetAccum);
    ImageMetrics budgetMetrics = compareImages(budgetImg, ref.image);
    saveImageBufferToPNG(budgetImg, "output/eq_pt_equal_time.png");

    std::cout << "\n  Equal-time result\n";
    char out[512];
    snprintf(out, sizeof(out),
             "    path tracer costs %.2f ms per sample per frame at %ux%u\n"
             "    in the real-time budget of %.2f ms it fits %.2f spp (rendered %u spp in %.2f ms)\n"
             "    equal-time path traced image: RMSE %.4f, PSNR %.2f dB, SSIM %.4f\n"
             "    real-time frame:              RMSE %.4f, PSNR %.2f dB, SSIM %.4f",
             msPerSpp, width, height, rcMs, rcMs / std::max(1e-9, msPerSpp), sppInBudget, budgetMs,
             budgetMetrics.rmse, budgetMetrics.psnr, budgetMetrics.ssim,
             rcMetrics.rmse, rcMetrics.psnr, rcMetrics.ssim);
    std::cout << out << "\n";

    if (crossoverSpp > 0) {
        double crossoverMs = double(crossoverSpp) * msPerSpp;
        snprintf(out, sizeof(out),
                 "    path tracing first matches the real-time RMSE at %u spp = %.1f ms, "
                 "%.0fx the real-time frame time",
                 crossoverSpp, crossoverMs, crossoverMs / std::max(1e-9, rcMs));
        std::cout << out << "\n";
    } else {
        snprintf(out, sizeof(out),
                 "    path tracing had not reached the real-time RMSE by %u spp (%.1f ms)",
                 maxSpp, elapsedMs);
        std::cout << out << "\n";
    }
    std::cout << "  wrote output/eq_reference.png, output/eq_realtime.png, output/eq_pt_equal_time.png\n";
    return 0;
}

// 3. What is each pass actually worth, in milliseconds and in error?
static int runAblationStudy(RendererState &state, uint32_t width, uint32_t height,
                            uint32_t refSpp, uint32_t chunk, uint32_t mode) {
    system("mkdir -p output");
    id<MTLTexture> target = makeReadbackTexture(state, width, height);

    std::cout << "\n=== Ablation ===\n";
    std::cout << "  " << width << "x" << height << ", mode " << mode
              << " (" << opticalModeName(mode) << "), error against a "
              << refSpp << " spp path traced reference\n\n";

    ReferenceResult ref = renderReference(state, target, mode, refSpp, chunk);
    saveImageBufferToPNG(ref.image, "output/abl_reference.png");

    struct AblationCase {
        uint32_t mask;
        const char *label;
        const char *tag;
    };
    const AblationCase cases[] = {
        { kAblAll,                        "full technique",     "full" },
        { kAblAll & ~kAblCascadeMerge,    "no cascade merge",   "nomerge" },
        { kAblAll & ~kAblCascadeGI,       "no cascade GI",      "nogi" },
        { kAblAll & ~kAblCaustics,        "no caustic splat",   "nocaustics" },
        { kAblAll & ~kAblAtlasFilter,     "no atlas filter",    "nofilter" },
        { kAblAll & ~kAblTemporal,        "no temporal blend",  "notemporal" },
        { kAblAll & ~kAblDispersion,      "no dispersion",      "nodispersion" },
    };

    char refLine[256];
    snprintf(refLine, sizeof(refLine),
             "  reference: %u spp in %.1f s, residual noise RMSE %.4f - differences below that are not measurable\n",
             ref.spp, ref.seconds, ref.noiseFloorRmse);
    std::cout << refLine << "\n";

    printMetricsHeader();

    std::vector<double> times;
    std::vector<RegionMetrics> metrics;
    for (const AblationCase &c : cases) {
        state.renderMode = mode;
        state.ablationMask = c.mask;
        double frameMs = timeFramesMs(state, target, 40, 20);
        ImageBuffer img = captureFrame(state, target);
        RegionMetrics m = compareAllRegions(img, ref);
        printMetricsRow(c.label, frameMs, m);

        std::string tag(c.tag);
        saveImageBufferToPNG(img, "output/abl_" + tag + ".png");
        saveErrorHeatmap(img, ref.image, "output/abl_" + tag + "_error.png", 0.25f);

        times.push_back(frameMs);
        metrics.push_back(m);
    }
    state.ablationMask = kAblAll;

    std::cout << "\n  What each pass is worth (removing it, against the full technique):\n";
    std::cout << "  case                   ms saved   dRMSE all  dRMSE floor  dRMSE glass   verdict\n";
    std::cout << "  ------------------------------------------------------------------------------\n";
    for (size_t i = 1; i < times.size(); i++) {
        double dMs = times[0] - times[i];
        double dAll = metrics[i].all.rmse - metrics[0].all.rmse;
        double dFloor = metrics[i].floor.rmse - metrics[0].floor.rmse;
        double dGlass = metrics[i].glass.rmse - metrics[0].glass.rmse;
        double biggest = std::max(std::fabs(dAll), std::max(std::fabs(dFloor), std::fabs(dGlass)));

        const char *verdict;
        if (biggest < ref.noiseFloorRmse) {
            verdict = "no measurable effect at this reference quality";
        } else if (dAll > 0.0 || dFloor > 0.0) {
            verdict = "removing it costs accuracy - the pass pays for itself";
        } else {
            verdict = "removing it is cheaper AND closer to ground truth";
        }

        char row[320];
        snprintf(row, sizeof(row), "  %-20s %9.2f  %+10.4f  %+11.4f  %+11.4f   %s",
                 cases[i].label, dMs, dAll, dFloor, dGlass, verdict);
        std::cout << row << "\n";
    }
    std::cout << "\n  wrote output/abl_reference.png and output/abl_<case>[_error].png\n";
    return 0;
}

int runHeadlessBenchmark(RendererState &state) {
    std::cout << "\n1920x1080, 20 frames per mode after 3 warm-up frames\n";

    system("mkdir -p output");

    const uint32_t width = 1920;
    const uint32_t height = 1080;

    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
#if TARGET_OS_OSX
    desc.storageMode = MTLStorageModeManaged;
#endif
    id<MTLTexture> targetTex = [state.device newTextureWithDescriptor:desc];

    struct ModeTest {
        uint32_t mode;
        std::string name;
        std::string filename;
    };

    std::vector<ModeTest> modes = {
        { 1, "clear glass", "rc_glass_scene.png" },
        { 2, "frosted glass", "rc_glass_frosted.png" },
        { 3, "high dispersion", "rc_glass_dispersion.png" },
        { 0, "whitted baseline", "rc_glass_whitted.png" }
    };

    for (const auto &test : modes) {
        state.renderMode = test.mode;

        for (int f = 0; f < 3; f++) {
            id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
            renderFrame(state, targetTex, cmd, 0.016f);
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        const int numFrames = 20;
        auto tStart = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < numFrames; f++) {
            id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
            renderFrame(state, targetTex, cmd, 0.016f);
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        double avgFrameMs = totalMs / double(numFrames);
        double fps = 1000.0 / avgFrameMs;

        id<MTLCommandBuffer> snapCmd = [state.commandQueue commandBuffer];
        renderFrame(state, targetTex, snapCmd, 0.016f);
        if (targetTex.storageMode == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [snapCmd blitCommandEncoder];
            [blit synchronizeResource:targetTex];
            [blit endEncoding];
        }
        [snapCmd commit];
        [snapCmd waitUntilCompleted];

        std::string outPath = "output/" + test.filename;
        saveTextureToPNG(targetTex, outPath);

        std::cout << "  mode " << test.mode << "  " << test.name
                  << "  " << avgFrameMs << " ms (" << fps << " fps)"
                  << "  -> " << outPath << "\n";
    }

    // Mode 4 is progressive, so it is timed per sample rather than per frame.
    {
        const uint32_t spp = 64;
        state.renderMode = 4;
        state.ptOpticsMode = 1;
        state.ptSamplesPerLaunch = 1;
        resetPathTraceAccumulation(state);

        auto tStart = std::chrono::high_resolution_clock::now();
        for (uint32_t s = 0; s < spp; s++) {
            id<MTLCommandBuffer> cmd = [state.commandQueue commandBuffer];
            renderFrame(state, targetTex, cmd, 0.016f);
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        id<MTLCommandBuffer> snapCmd = [state.commandQueue commandBuffer];
        if (targetTex.storageMode == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [snapCmd blitCommandEncoder];
            [blit synchronizeResource:targetTex];
            [blit endEncoding];
        }
        [snapCmd commit];
        [snapCmd waitUntilCompleted];

        std::string outPath = "output/rc_glass_pathtraced.png";
        saveTextureToPNG(targetTex, outPath);
        std::cout << "  mode 4  path traced reference  " << (totalMs / double(spp))
                  << " ms/spp, " << spp << " spp in " << (totalMs / 1000.0) << " s"
                  << "  -> " << outPath << "\n";
    }

    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        std::string shaderPath = resolveExistingPath({
            "Metal/shaders/RCGlassShaders.metal",
            "shaders/RCGlassShaders.metal",
            "RCGlassShaders.metal"
        });

        std::string teapotBinPath = resolveExistingPath({
            "assets/teapot.bin",
            "../assets/teapot.bin",
            "../../assets/teapot.bin",
            "teapot.bin"
        });

        bool headless = false;
        bool doCompare = false;
        bool doEqualTime = false;
        bool doAblation = false;
        uint32_t refSpp = 256;
        uint32_t maxSpp = 512;
        uint32_t sppChunk = 8;
        uint32_t studyMode = 1;
        uint32_t cmpWidth = 1280;
        uint32_t cmpHeight = 720;
        std::vector<uint32_t> compareModes = { 1, 2, 3, 0 };

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--headless" || arg == "--benchmark") {
                headless = true;
            } else if (arg == "--compare") {
                doCompare = true;
            } else if (arg == "--equal-time") {
                doEqualTime = true;
            } else if (arg == "--ablation") {
                doAblation = true;
            } else if (arg == "--ref-spp" && i + 1 < argc) {
                refSpp = uint32_t(std::max(2, atoi(argv[++i])));
            } else if (arg == "--max-spp" && i + 1 < argc) {
                maxSpp = uint32_t(std::max(1, atoi(argv[++i])));
            } else if (arg == "--spp-chunk" && i + 1 < argc) {
                sppChunk = uint32_t(std::max(1, atoi(argv[++i])));
            } else if (arg == "--mode" && i + 1 < argc) {
                studyMode = uint32_t(std::max(0, atoi(argv[++i])));
            } else if (arg == "--modes" && i + 1 < argc) {
                compareModes.clear();
                std::string list = argv[++i];
                size_t pos = 0;
                while (pos <= list.size()) {
                    size_t comma = list.find(',', pos);
                    std::string token = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    if (!token.empty()) compareModes.push_back(uint32_t(atoi(token.c_str())));
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
                if (compareModes.empty()) compareModes = { 1 };
            } else if (arg == "--res" && i + 1 < argc) {
                std::string res = argv[++i];
                size_t x = res.find('x');
                if (x != std::string::npos) {
                    cmpWidth = uint32_t(std::max(64, atoi(res.substr(0, x).c_str())));
                    cmpHeight = uint32_t(std::max(64, atoi(res.substr(x + 1).c_str())));
                }
            } else if (arg == "--glass-bounces" && i + 1 < argc) {
                gRenderer.glassBounces = uint32_t(std::min(8, std::max(1, atoi(argv[++i]))));
            } else if (arg == "--depth" && i + 1 < argc) {
                gRenderer.ptMaxDepth = uint32_t(std::max(1, atoi(argv[++i])));
            } else if (arg == "--sun-radius" && i + 1 < argc) {
                gRenderer.ptSunAngularRadiusDeg = float(atof(argv[++i]));
            } else if (arg == "--pt-clamp" && i + 1 < argc) {
                gRenderer.ptIndirectClamp = float(atof(argv[++i]));
            } else if (arg == "--teapot" && i + 1 < argc) {
                teapotBinPath = argv[++i];
            } else if (arg == "--shader" && i + 1 < argc) {
                shaderPath = argv[++i];
            }
        }

        if (!initMetal(gRenderer, shaderPath, teapotBinPath)) {
            return 1;
        }

        if (doCompare || doEqualTime || doAblation) {
            int rc = 0;
            if (doCompare)   rc |= runReferenceComparison(gRenderer, cmpWidth, cmpHeight, refSpp, sppChunk, compareModes);
            if (doEqualTime) rc |= runEqualTimeComparison(gRenderer, cmpWidth, cmpHeight, refSpp, sppChunk, maxSpp, studyMode);
            if (doAblation)  rc |= runAblationStudy(gRenderer, cmpWidth, cmpHeight, refSpp, sppChunk, studyMode);
            return rc;
        }

        if (headless) {
            return runHeadlessBenchmark(gRenderer);
        }

        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        AppDelegate *delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];

        [app run];
    }
    return 0;
}
