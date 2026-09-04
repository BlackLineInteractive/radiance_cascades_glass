#define USE_GLM
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "gl_loader.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TEAPOT_DATA_NO_SIMD
#include "TeapotData.h"
#include "Camera.h"

struct alignas(16) GlassUniformsGL {
    glm::mat4 viewInverse;
    glm::mat4 projectionInverse;
    glm::vec3 cameraPosition;
    float time;

    glm::vec3 sunDirection;
    float sunIntensity;
    glm::vec3 sunColor;
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
    glm::vec2 pad;
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

static bool saveGLTextureToPNG(GLuint texture, uint32_t w, uint32_t h, const std::string &outPNGPath) {
    std::vector<float> floatPixels(w * h * 4);
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, floatPixels.data());

    std::vector<uint8_t> bytePixels(w * h * 4);
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            size_t srcIdx = (y * w + x) * 4;
            size_t dstIdx = ((h - 1 - y) * w + x) * 4;
            bytePixels[dstIdx + 0] = static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, floatPixels[srcIdx + 0])) * 255.0f);
            bytePixels[dstIdx + 1] = static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, floatPixels[srcIdx + 1])) * 255.0f);
            bytePixels[dstIdx + 2] = static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, floatPixels[srcIdx + 2])) * 255.0f);
            bytePixels[dstIdx + 3] = 255;
        }
    }

    std::string tmpTGA = outPNGPath + ".tmp.tga";
    saveTGA(tmpTGA, w, h, bytePixels);
    std::string sipsCmd = "sips -s format png " + tmpTGA + " --out " + outPNGPath + " > /dev/null 2>&1 && rm -f " + tmpTGA;
    int ret = system(sipsCmd.c_str());
    return (ret == 0);
}

static std::string readFile(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::string resolveExistingPath(const std::vector<std::string> &candidates) {
    for (const auto &c : candidates) {
        std::ifstream f(c.c_str());
        if (f.good()) return c;
    }
    return candidates.empty() ? "" : candidates[0];
}

static GLuint compileComputeShader(const std::string &shaderPath) {
    std::string src = readFile(shaderPath);
    if (src.empty()) {
        std::cerr << "[OpenGL] Failed to read shader file: " << shaderPath << "\n";
        return 0;
    }

    // Resolve #include directives
    size_t incPos = 0;
    while ((incPos = src.find("#include \"", incPos)) != std::string::npos) {
        size_t start = incPos + 10;
        size_t end = src.find("\"", start);
        if (end == std::string::npos) break;
        std::string incFile = src.substr(start, end - start);
        std::string incPath = resolveExistingPath({
            incFile,
            "OpenGL/shaders/" + incFile,
            "shaders/" + incFile,
            "common/" + incFile,
            "../common/" + incFile,
            "../../common/" + incFile
        });
        std::string incSrc = readFile(incPath);
        src.replace(incPos, end - incPos + 1, incSrc);
        incPos += incSrc.length();
    }

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char *cSrc = src.c_str();
    glShaderSource(shader, 1, &cSrc, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "[OpenGL] Shader compilation error in " << shaderPath << ":\n" << infoLog << "\n";
        glDeleteShader(shader);
        return 0;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);

    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(prog, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "[OpenGL] Program link error in " << shaderPath << ":\n" << infoLog << "\n";
        glDeleteProgram(prog);
        glDeleteShader(shader);
        return 0;
    }

    glDeleteShader(shader);
    return prog;
}

struct OpenGLRenderer {
    GLFWwindow *window = nullptr;

    GLuint cascadeProgram = 0;
    GLuint filterAtlasProgram = 0;
    GLuint causticsGenProgram = 0;
    GLuint causticsFilterProgram = 0;
    GLuint sceneProgram = 0;
    GLuint quadProgram = 0;

    GLuint irradianceAtlas = 0;
    GLuint filteredIrradianceAtlas = 0;
    GLuint causticTexture = 0;
    GLuint outTexture = 0;

    GLuint uniformBuffer = 0;
    GLuint bvhBuffer = 0;
    GLuint triBuffer = 0;
    GLuint causticBuffer = 0;
    GLuint quadVAO = 0;

    uint32_t numTeapotNodes = 0;
    uint32_t numTeapotTris = 0;
    uint32_t width = 1280;
    uint32_t height = 720;

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

static OpenGLRenderer gGL;

static GLuint createStorageTexture(uint32_t w, uint32_t h) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static GLuint compileQuadProgram() {
    const char *vsSrc = R"(#version 330 core
        void main() {
            vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
        }
    )";

    const char *fsSrc = R"(#version 330 core
        out vec4 fragColor;
        uniform sampler2D tex;
        void main() {
            ivec2 sz = textureSize(tex, 0);
            vec2 uv = gl_FragCoord.xy / vec2(sz);
            fragColor = texture(tex, uv);
        }
    )";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSrc, nullptr);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, nullptr);
    glCompileShader(fs);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

bool initOpenGL(OpenGLRenderer &r, const std::string &teapotBinPath, bool headless) {
    if (!glfwInit()) {
        std::cerr << "[OpenGL] Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (headless) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    r.window = glfwCreateWindow(r.width, r.height, "Radiance Cascades Glass & Caustics [OpenGL 4.3]", nullptr, nullptr);
    if (!r.window) {
        std::cerr << "[OpenGL] Warning: Failed to create OpenGL 4.3 Core context via GLFW.\n";
        std::cerr << "         Note: Native Apple macOS drivers are capped at OpenGL 4.1.\n";
        std::cerr << "         Use the Metal backend (./Metal/build.sh) or Vulkan backend (./Vulkan/build.sh) on macOS.\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(r.window);
    glfwSwapInterval(0);

    if (!initGLLoader()) {
        std::cerr << "[OpenGL] Failed to load required OpenGL 4.3 Core function pointers\n";
        return false;
    }

    std::cout << "[OpenGL] Driver: " << glGetString(GL_RENDERER) << " (" << glGetString(GL_VERSION) << ")\n";

    std::string shaderDir = resolveExistingPath({
        "OpenGL/shaders/radiance_cascades.comp",
        "shaders/radiance_cascades.comp"
    });
    std::string baseDir = shaderDir.substr(0, shaderDir.find_last_of("/\\") + 1);

    r.cascadeProgram        = compileComputeShader(baseDir + "radiance_cascades.comp");
    r.filterAtlasProgram    = compileComputeShader(baseDir + "filter_atlas.comp");
    r.causticsGenProgram    = compileComputeShader(baseDir + "caustics_generate.comp");
    r.causticsFilterProgram = compileComputeShader(baseDir + "caustics_filter.comp");
    r.sceneProgram          = compileComputeShader(baseDir + "render_scene.comp");
    r.quadProgram           = compileQuadProgram();

    if (!r.cascadeProgram || !r.filterAtlasProgram || !r.causticsGenProgram || !r.causticsFilterProgram || !r.sceneProgram) {
        std::cerr << "[OpenGL] Fatal: Failed to compile one or more compute shaders\n";
        return false;
    }

    r.irradianceAtlas         = createStorageTexture(320, 64);
    r.filteredIrradianceAtlas = createStorageTexture(320, 64);
    r.causticTexture          = createStorageTexture(1024, 1024);
    r.outTexture              = createStorageTexture(r.width, r.height);

    glGenBuffers(1, &r.uniformBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, r.uniformBuffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GlassUniformsGL), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, r.uniformBuffer);

    TeapotMesh teapot;
    if (teapot.loadFromBinary(teapotBinPath)) {
        r.numTeapotNodes = static_cast<uint32_t>(teapot.nodes.size());
        r.numTeapotTris  = static_cast<uint32_t>(teapot.triangles.size());

        glGenBuffers(1, &r.bvhBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, r.bvhBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, teapot.nodes.size() * sizeof(GPUBVHNode), teapot.nodes.data(), GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, r.bvhBuffer);

        glGenBuffers(1, &r.triBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, r.triBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, teapot.triangles.size() * sizeof(GPUTriangle), teapot.triangles.data(), GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, r.triBuffer);
    }

    size_t causticSize = 1024 * 1024 * 4 * sizeof(uint32_t);
    glGenBuffers(1, &r.causticBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, r.causticBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, causticSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, r.causticBuffer);

    glGenVertexArrays(1, &r.quadVAO);

    return true;
}

void renderFrameGL(OpenGLRenderer &r, float deltaTime) {
    if (!r.sunPaused) {
        r.sunTime += deltaTime * r.sunSpeed;
    }

    float sunAzimuth = r.sunTime * 0.35f + r.manualAzimuthOffset;
    float sunX = -0.78f + std::sin(sunAzimuth) * 0.12f;
    float sunY =  0.55f + std::cos(sunAzimuth * 0.7f) * 0.10f + r.manualElevationOffset;
    float sunZ =  std::cos(sunAzimuth) * 0.42f + 0.05f;
    glm::vec3 sunDir = glm::normalize(glm::vec3(sunX, sunY, sunZ));

    GlassUniformsGL uniforms;
    r.camera.aspect = float(r.width) / float(r.height);
    glm::mat4 viewMat = r.camera.getViewMatrix();
    glm::mat4 projMat = r.camera.getProjectionMatrix();

    uniforms.viewInverse = glm::inverse(viewMat);
    uniforms.projectionInverse = glm::inverse(projMat);
    uniforms.cameraPosition = r.camera.getPosition();
    uniforms.time = r.sunTime;

    uniforms.sunDirection = sunDir;
    uniforms.sunIntensity = 2.8f;
    uniforms.sunColor = glm::vec3(1.0f, 0.98f, 0.92f);
    uniforms.ambientIntensity = 0.25f;

    uniforms.glassIor = 1.52f;
    uniforms.glassDispersion = 0.025f;
    uniforms.glassRoughness = r.glassRoughness;
    uniforms.glassAbsorption = 0.05f;

    uniforms.renderMode = r.renderMode;
    uniforms.width = r.width;
    uniforms.height = r.height;
    uniforms.frameIndex = r.frameCount++;

    uniforms.numTeapotNodes = r.numTeapotNodes;
    uniforms.numTeapotTris  = r.numTeapotTris;
    uniforms.pad = glm::vec2(0.0f);

    glBindBuffer(GL_UNIFORM_BUFFER, r.uniformBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(GlassUniformsGL), &uniforms);

    if (r.renderMode != 0) {
        glUseProgram(r.cascadeProgram);
        glBindImageTexture(4, r.irradianceAtlas, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute((320 + 15) / 16, (64 + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        glUseProgram(r.filterAtlasProgram);
        glBindImageTexture(4, r.irradianceAtlas, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(5, r.filteredIrradianceAtlas, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute((320 + 15) / 16, (64 + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        size_t causticSize = 1024 * 1024 * 4 * sizeof(uint32_t);
        static std::vector<uint32_t> zeroBuffer(1024 * 1024 * 4, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, r.causticBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, causticSize, zeroBuffer.data());

        glUseProgram(r.causticsGenProgram);
        glDispatchCompute((2048 + 15) / 16, (2048 + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glUseProgram(r.causticsFilterProgram);
        glBindImageTexture(6, r.causticTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute((1024 + 15) / 16, (1024 + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    }

    glUseProgram(r.sceneProgram);
    glBindImageTexture(7, r.outTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, r.renderMode != 0 ? r.filteredIrradianceAtlas : r.irradianceAtlas);
    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, r.causticTexture);

    glDispatchCompute((r.width + 15) / 16, (r.height + 15) / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

int runHeadlessGL(OpenGLRenderer &r) {
    std::cout << "\n=================================================================\n";
    std::cout << "  Radiance Cascades Glass: OpenGL Automated Benchmark\n";
    std::cout << "=================================================================\n";

    system("mkdir -p output");

    struct ModeTest {
        uint32_t mode;
        std::string name;
        std::string filename;
        std::string description;
    };

    std::vector<ModeTest> modes = {
        { 1, "Mode 1: Realistic Glass + Caustics", "rc_glass_gl_scene.png", "Snell refraction + Fresnel + Cauchy dispersion + floor caustics" },
        { 2, "Mode 2: Frosted / Rough Glass Cascade", "rc_glass_gl_frosted.png", "Micro-roughness transmission cone + diffused caustic filter" },
        { 3, "Mode 3: High Spectral Dispersion Prism", "rc_glass_gl_dispersion.png", "Amplified dispersion on Newton's prism & crystal sphere" },
        { 0, "Mode 0: Whitted RT Baseline", "rc_glass_gl_whitted.png", "Classic binary shadow ray (zero caustics, dark shadow)" }
    };

    for (const auto &test : modes) {
        r.renderMode = test.mode;

        for (int f = 0; f < 3; f++) {
            renderFrameGL(r, 0.016f);
            glFinish();
        }

        const int numFrames = 20;
        auto tStart = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < numFrames; f++) {
            renderFrameGL(r, 0.016f);
            glFinish();
        }
        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        double avgFrameMs = totalMs / double(numFrames);
        double fps = 1000.0 / avgFrameMs;

        std::string outPath = "output/" + test.filename;
        saveGLTextureToPNG(r.outTexture, r.width, r.height, outPath);

        std::cout << ">>> " << test.name << "\n";
        std::cout << "    " << test.description << "\n";
        std::cout << "    [Perf] " << avgFrameMs << " ms (" << fps << " FPS)\n";
        std::cout << "    [Output] " << outPath << "\n\n";
    }

    std::cout << "OpenGL Benchmark complete.\n";
    return 0;
}

static double sLastX = 0.0, sLastY = 0.0;
static bool sLeftPressed = false, sRightPressed = false;

static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        sLeftPressed = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        sRightPressed = (action == GLFW_PRESS);
    }
}

static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
    double dx = xpos - sLastX;
    double dy = ypos - sLastY;
    sLastX = xpos;
    sLastY = ypos;

    if (sLeftPressed) {
        if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) {
            gGL.camera.pan(static_cast<float>(dx), static_cast<float>(dy));
        } else {
            gGL.camera.orbit(static_cast<float>(dx), static_cast<float>(dy));
        }
    } else if (sRightPressed) {
        gGL.camera.zoom(static_cast<float>(dy) * 0.03f);
    }
}

static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    gGL.camera.zoom(static_cast<float>(yoffset) * 0.25f);
}

static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    switch (key) {
        case GLFW_KEY_SPACE:
            gGL.sunPaused = !gGL.sunPaused;
            std::cout << "[Sun] Dynamic motion: " << (gGL.sunPaused ? "PAUSED" : "RESUMED") << "\n";
            break;
        case GLFW_KEY_1:
            gGL.renderMode = 1;
            std::cout << "[Mode] Mode 1: Clear Glass + Radiance Cascades\n";
            break;
        case GLFW_KEY_2:
            gGL.renderMode = 2;
            std::cout << "[Mode] Mode 2: Frosted Rough Glass\n";
            break;
        case GLFW_KEY_3:
            gGL.renderMode = 3;
            std::cout << "[Mode] Mode 3: High Spectral Dispersion\n";
            break;
        case GLFW_KEY_0:
            gGL.renderMode = 0;
            std::cout << "[Mode] Mode 0: Whitted RT Baseline\n";
            break;
        case GLFW_KEY_R:
            gGL.camera = OrbitCamera();
            std::cout << "[Camera] Reset view\n";
            break;
        case GLFW_KEY_EQUAL:
            gGL.glassRoughness = std::min(1.0f, gGL.glassRoughness + 0.05f);
            std::cout << "[Roughness] " << gGL.glassRoughness << "\n";
            break;
        case GLFW_KEY_MINUS:
            gGL.glassRoughness = std::max(0.0f, gGL.glassRoughness - 0.05f);
            std::cout << "[Roughness] " << gGL.glassRoughness << "\n";
            break;
        case GLFW_KEY_S: {
            system("mkdir -p output");
            std::string outPath = "output/rc_glass_gl_snapshot.png";
            if (saveGLTextureToPNG(gGL.outTexture, gGL.width, gGL.height, outPath)) {
                std::cout << "[Capture] Screenshot saved to " << outPath << "\n";
            }
            break;
        }
        case GLFW_KEY_LEFT:
            gGL.manualAzimuthOffset -= 0.08f;
            break;
        case GLFW_KEY_RIGHT:
            gGL.manualAzimuthOffset += 0.08f;
            break;
        case GLFW_KEY_UP:
            gGL.manualElevationOffset += 0.04f;
            break;
        case GLFW_KEY_DOWN:
            gGL.manualElevationOffset -= 0.04f;
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        default:
            break;
    }
}

int main(int argc, const char *argv[]) {
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
        }
    }

    if (!initOpenGL(gGL, teapotBinPath, headless)) {
        return 1;
    }

    if (headless) {
        int res = runHeadlessGL(gGL);
        glfwDestroyWindow(gGL.window);
        glfwTerminate();
        return res;
    }

    glfwSetMouseButtonCallback(gGL.window, mouseButtonCallback);
    glfwSetCursorPosCallback(gGL.window, cursorPosCallback);
    glfwSetScrollCallback(gGL.window, scrollCallback);
    glfwSetKeyCallback(gGL.window, keyCallback);

    std::cout << "=================================================================\n";
    std::cout << "  Radiance Cascades Glass & Caustics Engine [OpenGL 4.3]\n";
    std::cout << "=================================================================\n";
    std::cout << "  Controls:\n";
    std::cout << "    [Left Mouse Drag]   : Orbit Camera\n";
    std::cout << "    [Alt + Drag]        : Pan Camera\n";
    std::cout << "    [Right Drag/Scroll] : Zoom Camera\n";
    std::cout << "    [Space]             : Toggle Sun Animation\n";
    std::cout << "    [Arrow Keys]        : Adjust Sun Position\n";
    std::cout << "    [1]                 : Clear Glass Mode\n";
    std::cout << "    [2]                 : Frosted Glass Mode\n";
    std::cout << "    [3]                 : High Dispersion Prism Mode\n";
    std::cout << "    [0]                 : Whitted Ray Tracing Baseline\n";
    std::cout << "    [+/-]               : Adjust Roughness\n";
    std::cout << "    [R]                 : Reset Camera\n";
    std::cout << "    [S]                 : Screenshot\n";
    std::cout << "    [ESC]               : Quit\n";
    std::cout << "=================================================================\n\n";

    gGL.lastFrameTime = glfwGetTime();

    while (!glfwWindowShouldClose(gGL.window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        double dt = now - gGL.lastFrameTime;
        gGL.lastFrameTime = now;
        if (dt > 0.0) {
            double curFps = 1.0 / dt;
            gGL.fps = gGL.fps * 0.9 + curFps * 0.1;
        }

        renderFrameGL(gGL, static_cast<float>(dt));

        glViewport(0, 0, gGL.width, gGL.height);
        glUseProgram(gGL.quadProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gGL.outTexture);
        glBindVertexArray(gGL.quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(gGL.window);

        if (gGL.frameCount % 60 == 0) {
            char title[256];
            snprintf(title, sizeof(title), "Radiance Cascades Glass [OpenGL 4.3] | Mode: %u | FPS: %.1f", gGL.renderMode, gGL.fps);
            glfwSetWindowTitle(gGL.window, title);
        }
    }

    glfwDestroyWindow(gGL.window);
    glfwTerminate();
    return 0;
}
