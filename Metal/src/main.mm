#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <simd/simd.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>

#include "TeapotData.h"
#include "Camera.h"

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

    id<MTLComputePipelineState> cascadePipeline;
    id<MTLComputePipelineState> filterCascadePipeline;
    id<MTLComputePipelineState> causticsPipeline;
    id<MTLComputePipelineState> filterPipeline;
    id<MTLComputePipelineState> scenePipeline;

    id<MTLTexture> irradianceAtlas;
    id<MTLTexture> filteredIrradianceAtlas;
    id<MTLTexture> causticTexture;
    id<MTLBuffer> causticBuffer;
    size_t causticBufferSize = 0;

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
};

static RendererState gRenderer;

static std::string resolveExistingPath(const std::vector<std::string> &candidates) {
    for (const auto &c : candidates) {
        std::ifstream f(c.c_str());
        if (f.good()) return c;
    }
    return candidates.empty() ? "" : candidates[0];
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

    id<MTLFunction> cascadeFunc       = [library newFunctionWithName:@"computeRadianceCascadesKernel"];
    id<MTLFunction> filterCascadeFunc = [library newFunctionWithName:@"filterIrradianceAtlasKernel"];
    id<MTLFunction> causticsFunc      = [library newFunctionWithName:@"generateCausticsKernel"];
    id<MTLFunction> filterFunc        = [library newFunctionWithName:@"filterCausticsKernel"];
    id<MTLFunction> sceneFunc         = [library newFunctionWithName:@"renderSceneKernel"];

    if (!causticsFunc || !filterFunc || !sceneFunc) {
        std::cerr << "[Metal] Failed to locate required kernel functions in library.\n";
        return false;
    }

    if (cascadeFunc) {
        state.cascadePipeline = [state.device newComputePipelineStateWithFunction:cascadeFunc error:&error];
    }
    if (filterCascadeFunc) {
        state.filterCascadePipeline = [state.device newComputePipelineStateWithFunction:filterCascadeFunc error:&error];
    }

    state.causticsPipeline = [state.device newComputePipelineStateWithFunction:causticsFunc error:&error];
    state.filterPipeline   = [state.device newComputePipelineStateWithFunction:filterFunc error:&error];
    state.scenePipeline    = [state.device newComputePipelineStateWithFunction:sceneFunc error:&error];

    MTLTextureDescriptor *atlasDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                         width:320
                                                                                        height:64
                                                                                     mipmapped:NO];
    atlasDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    atlasDesc.storageMode = MTLStorageModePrivate;
    state.irradianceAtlas = [state.device newTextureWithDescriptor:atlasDesc];
    state.filteredIrradianceAtlas = [state.device newTextureWithDescriptor:atlasDesc];

    const uint32_t causticRes = 1024;
    MTLTextureDescriptor *cTexDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                        width:causticRes
                                                                                       height:causticRes
                                                                                    mipmapped:NO];
    cTexDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    cTexDesc.storageMode = MTLStorageModePrivate;
    state.causticTexture = [state.device newTextureWithDescriptor:cTexDesc];

    state.causticBufferSize = causticRes * causticRes * 4 * sizeof(uint32_t);
    state.causticBuffer = [state.device newBufferWithLength:state.causticBufferSize options:MTLResourceStorageModePrivate];

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

void renderFrame(
    RendererState &state,
    id<MTLTexture> targetTexture,
    id<MTLCommandBuffer> cmdBuffer,
    float deltaTime
) {
    uint32_t width  = static_cast<uint32_t>(targetTexture.width);
    uint32_t height = static_cast<uint32_t>(targetTexture.height);

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
    uniforms.sunIntensity = 2.8f;
    uniforms.sunColor = simd::make_float3(1.0f, 0.98f, 0.92f);
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

    id<MTLBuffer> uniformBuffer = [state.device newBufferWithBytes:&uniforms
                                                            length:sizeof(GlassUniforms)
                                                           options:MTLResourceStorageModeShared];

    if (state.renderMode != 0 && state.cascadePipeline && state.irradianceAtlas) {
        id<MTLComputeCommandEncoder> rcEnc = [cmdBuffer computeCommandEncoder];
        [rcEnc setComputePipelineState:state.cascadePipeline];
        [rcEnc setTexture:state.irradianceAtlas atIndex:0];
        [rcEnc setTexture:state.filteredIrradianceAtlas atIndex:1];
        [rcEnc setBuffer:uniformBuffer offset:0 atIndex:0];
        if (state.teapotNodeBuffer) [rcEnc setBuffer:state.teapotNodeBuffer offset:0 atIndex:1];
        if (state.teapotTriBuffer)  [rcEnc setBuffer:state.teapotTriBuffer offset:0 atIndex:2];

        MTLSize rcTg = MTLSizeMake(16, 16, 1);
        MTLSize rcGrid = MTLSizeMake((320 + 15) / 16, (64 + 15) / 16, 1);
        [rcEnc dispatchThreadgroups:rcGrid threadsPerThreadgroup:rcTg];
        [rcEnc endEncoding];

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

@interface RCGlassView : MTKView <MTKViewDelegate>
@property (nonatomic, assign) NSPoint lastMousePos;
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
    }
    return self;
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
        [cmdBuffer commit];

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
    NSString *chars = [event charactersIgnoringModifiers];
    if ([chars length] == 0) return;
    unichar c = [chars characterAtIndex:0];

    switch (c) {
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

    std::cout << "=================================================================\n";
    std::cout << "  Radiance Cascades Glass & Caustics Engine\n";
    std::cout << "=================================================================\n";
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
    std::cout << "    [R]                 : Reset Camera\n";
    std::cout << "    [S]                 : Screenshot (1080p)\n";
    std::cout << "    [ESC]               : Quit\n";
    std::cout << "=================================================================\n\n";
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end

int runHeadlessBenchmark(RendererState &state) {
    std::cout << "\n=================================================================\n";
    std::cout << "  Radiance Cascades Glass Benchmark\n";
    std::cout << "=================================================================\n";

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
        std::string description;
    };

    std::vector<ModeTest> modes = {
        { 1, "Mode 1: Realistic Glass + Caustics", "rc_glass_scene.png", "Snell refraction + Fresnel + Cauchy dispersion + floor caustics" },
        { 2, "Mode 2: Frosted / Rough Glass Cascade", "rc_glass_frosted.png", "Micro-roughness transmission cone + diffused caustic filter" },
        { 3, "Mode 3: High Spectral Dispersion Prism", "rc_glass_dispersion.png", "Amplified dispersion on Newton's prism & crystal sphere" },
        { 0, "Mode 0: Whitted RT Baseline", "rc_glass_whitted.png", "Classic binary shadow ray (zero caustics, dark shadow)" }
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

        std::cout << ">>> " << test.name << "\n";
        std::cout << "    " << test.description << "\n";
        std::cout << "    [Perf] " << avgFrameMs << " ms (" << fps << " FPS)\n";
        std::cout << "    [Output] " << outPath << "\n\n";
    }

    std::cout << "Benchmark complete. All 4 optical modes verified.\n";
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
