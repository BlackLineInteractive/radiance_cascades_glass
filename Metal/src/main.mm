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
    simd::float2 pad;
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

    id<MTLTexture> irradianceAtlas;
    id<MTLTexture> filteredIrradianceAtlas;
    id<MTLTexture> cascadeTex[4];
    id<MTLTexture> dummyCascadeTexture;
    id<MTLBuffer> cascadeParamsBuffer[4];
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

    if (!cascadeGatherFunc || !cascadeIntegrateFunc || !filterCascadeFunc || !causticsFunc || !filterFunc || !sceneFunc) {
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

    if (!state.cascadeGatherPipeline || !state.cascadeIntegratePipeline || !state.filterCascadePipeline ||
        !state.causticsPipeline || !state.filterPipeline || !state.scenePipeline) {
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
    uniforms.pad = simd::make_float2(0.0f);

    id<MTLBuffer> uniformBuffer = state.uniformBuffer;
    memcpy(uniformBuffer.contents, &uniforms, sizeof(GlassUniforms));

    MTLSize rcTg = MTLSizeMake(16, 16, 1);
    MTLSize rcGrid = MTLSizeMake((320 + 15) / 16, (64 + 15) / 16, 1);

    if (state.renderMode != 0 && state.cascadeGatherPipeline && state.irradianceAtlas) {
        // Far-to-near: level 3 has nothing above it to read, level 0 reads level 1,
        // and so on down. Each dispatch covers exactly that level's probe x ray grid.
        for (int level = 3; level >= 0; level--) {
            const CascadeLevelSpec &spec = kCascadeLevelSpecs[level];
            id<MTLTexture> upperTex = (level == 3) ? state.dummyCascadeTexture : state.cascadeTex[level + 1];

            id<MTLComputeCommandEncoder> cgEnc = [cmdBuffer computeCommandEncoder];
            [cgEnc setComputePipelineState:state.cascadeGatherPipeline];
            [cgEnc setTexture:state.cascadeTex[level] atIndex:0];
            [cgEnc setTexture:upperTex atIndex:1];
            [cgEnc setTexture:state.filteredIrradianceAtlas atIndex:2];
            [cgEnc setBuffer:uniformBuffer offset:0 atIndex:0];
            [cgEnc setBuffer:state.cascadeParamsBuffer[level] offset:0 atIndex:1];
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

        if (state.filterCascadePipeline && state.filteredIrradianceAtlas) {
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

    if (state.renderMode != 0) {
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

    id<MTLComputeCommandEncoder> fEnc = [cmdBuffer computeCommandEncoder];
    [fEnc setComputePipelineState:state.filterPipeline];
    [fEnc setBuffer:state.causticBuffer offset:0 atIndex:0];
    [fEnc setTexture:state.causticTexture atIndex:0];
    [fEnc setBuffer:uniformBuffer offset:0 atIndex:1];

    MTLSize fTg = MTLSizeMake(16, 16, 1);
    MTLSize fGrid = MTLSizeMake((1024 + 15) / 16, (1024 + 15) / 16, 1);
    [fEnc dispatchThreadgroups:fGrid threadsPerThreadgroup:fTg];
    [fEnc endEncoding];

    id<MTLComputeCommandEncoder> rEnc = [cmdBuffer computeCommandEncoder];
    [rEnc setComputePipelineState:state.scenePipeline];
    [rEnc setTexture:targetTexture atIndex:0];
    [rEnc setTexture:state.causticTexture atIndex:1];
    id<MTLTexture> atlasToSample = state.filteredIrradianceAtlas ? state.filteredIrradianceAtlas : state.irradianceAtlas;
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
        "Mode 3: High Dispersion Prism"
    };
    const char *modeStr = (state.renderMode <= 3) ? modeNames[state.renderMode] : "Custom";

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
        @"Roughness:   %.2f\n"
        @"Brightness:  Mode %u/6 (%.1fx)\n"
        @"Light Color: %s",
        state.currentWidth, state.currentHeight,
        state.device ? [state.device name] : @"Metal Device",
        getGpuTypeDescription(state.device),
        effectiveGpuLoad, state.gpuDurationMs,
        state.fps, frameTimeMs,
        m.appVramMB, m.totalVramMB,
        m.procRamMB, m.sysUsedRamGB, m.sysTotalRamGB,
        modeStr, state.glassRoughness,
        (state.brightnessMode % 6) + 1, curBrightness,
        colorStr
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

        NSRect overlayFrame = NSMakeRect(16, frameRect.size.height - 195 - 16, 390, 195);
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
}

- (void)rightMouseDragged:(NSEvent *)event {
    NSPoint currentPos = [event locationInWindow];
    float dy = currentPos.y - self.lastMousePos.y;
    self.lastMousePos = currentPos;
    gRenderer.camera.zoom(dy * 0.03f);
}

- (void)scrollWheel:(NSEvent *)event {
    float delta = [event scrollingDeltaY];
    gRenderer.camera.zoom(delta * 0.05f);
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
        case '0':
            gRenderer.renderMode = 0;
            std::cout << "[Mode] Mode 0: Whitted RT Baseline\n";
            break;
        case 'r':
        case 'R':
            gRenderer.camera = OrbitCamera();
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
            break;
        case NSRightArrowFunctionKey:
            gRenderer.manualAzimuthOffset += 0.08f;
            break;
        case NSUpArrowFunctionKey:
            gRenderer.manualElevationOffset += 0.04f;
            break;
        case NSDownArrowFunctionKey:
            gRenderer.manualElevationOffset -= 0.04f;
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
    std::cout << "    [+/-]               : Adjust Roughness\n";
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
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--headless" || arg == "--benchmark") {
                headless = true;
            } else if (arg == "--teapot" && i + 1 < argc) {
                teapotBinPath = argv[++i];
            } else if (arg == "--shader" && i + 1 < argc) {
                shaderPath = argv[++i];
            }
        }

        if (!initMetal(gRenderer, shaderPath, teapotBinPath)) {
            return 1;
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
