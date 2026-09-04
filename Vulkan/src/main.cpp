#define USE_GLM
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TEAPOT_DATA_NO_SIMD
#include "TeapotData.h"
#include "Camera.h"

struct alignas(16) GlassUniformsVK {
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

static bool saveFloatBufferToPNG(const float *floatPixels, uint32_t w, uint32_t h, const std::string &outPNGPath) {
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

static std::vector<char> readSpvFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) return {};
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

static std::string resolveExistingPath(const std::vector<std::string> &candidates) {
    for (const auto &c : candidates) {
        std::ifstream f(c.c_str());
        if (f.good()) return c;
    }
    return candidates.empty() ? "" : candidates[0];
}

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct VulkanImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct VulkanRenderer {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline cascadeGatherPipeline[4] = {}; // index by cascade level 0..3
    VkPipeline cascadeIntegratePipeline = VK_NULL_HANDLE;
    VkPipeline filterAtlasPipeline = VK_NULL_HANDLE;
    VkPipeline causticsGenPipeline = VK_NULL_HANDLE;
    VkPipeline causticsFilterPipeline = VK_NULL_HANDLE;
    VkPipeline scenePipeline = VK_NULL_HANDLE;

    VkSampler linearSampler = VK_NULL_HANDLE;

    VulkanBuffer uniformBuffer;
    VulkanBuffer bvhBuffer;
    VulkanBuffer triBuffer;
    VulkanBuffer causticBuffer;
    VulkanBuffer stagingBuffer;

    VulkanImage irradianceAtlas;
    VulkanImage filteredAtlas;
    VulkanImage cascadeTex[4]; // one per cascade level, 5 array layers (rooms surfaces) each
    VulkanImage causticTexture;
    VulkanImage outTexture;

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
};

static VulkanRenderer gVK;

static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

static bool createBuffer(VulkanRenderer &r, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBuffer &buf) {
    buf.size = size;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(r.device, &bufferInfo, nullptr, &buf.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(r.device, buf.buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(r.physicalDevice, memReq.memoryTypeBits, properties);

    if (vkAllocateMemory(r.device, &allocInfo, nullptr, &buf.memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(r.device, buf.buffer, buf.memory, 0);
    return true;
}

static bool createImage(VulkanRenderer &r, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VulkanImage &img) {
    img.width = w;
    img.height = h;
    img.format = format;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = w;
    imageInfo.extent.height = h;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(r.device, &imageInfo, nullptr, &img.image) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(r.device, img.image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(r.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(r.device, &allocInfo, nullptr, &img.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(r.device, img.image, img.memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = img.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(r.device, &viewInfo, nullptr, &img.view) != VK_SUCCESS) return false;

    return true;
}

// A cascade level's probe grid, one array layer per room surface. Kept
// separate from createImage because array views need VK_IMAGE_VIEW_TYPE_2D_ARRAY
// and a layerCount to match, not just a different arrayLayers count.
static bool createImageArray(VulkanRenderer &r, uint32_t w, uint32_t h, uint32_t layers, VkFormat format, VkImageUsageFlags usage, VulkanImage &img) {
    img.width = w;
    img.height = h;
    img.format = format;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = w;
    imageInfo.extent.height = h;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layers;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(r.device, &imageInfo, nullptr, &img.image) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(r.device, img.image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(r.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(r.device, &allocInfo, nullptr, &img.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(r.device, img.image, img.memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = img.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layers;

    if (vkCreateImageView(r.device, &viewInfo, nullptr, &img.view) != VK_SUCCESS) return false;

    return true;
}

static VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string &spvPath) {
    std::vector<char> code = readSpvFile(spvPath);
    if (code.empty()) {
        std::cerr << "[Vulkan] Failed to load SPIR-V shader: " << spvPath << "\n";
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = layout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan] vkCreateComputePipelines failed with code " << res << " for: " << spvPath << "\n";
        pipeline = VK_NULL_HANDLE;
    }

    vkDestroyShaderModule(device, shaderModule, nullptr);
    return pipeline;
}

static void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool initVulkan(VulkanRenderer &r, const std::string &teapotBinPath) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Radiance Cascades Glass Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "RC_Glass";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> instanceExtensions;
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availableExts.data());

    bool hasPortabilityEnum = false;
    for (const auto &ext : availableExts) {
        if (strcmp(ext.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            hasPortabilityEnum = true;
            instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            break;
        }
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (hasPortabilityEnum) {
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &r.instance) != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create Vulkan instance\n";
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(r.instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[Vulkan] No physical GPU device found supporting Vulkan\n";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(r.instance, &deviceCount, devices.data());

    r.physicalDevice = devices[0];
    for (const auto &dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            r.physicalDevice = dev;
            break;
        }
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(r.physicalDevice, &props);
    std::cout << "[Vulkan] GPU Device: " << props.deviceName << "\n";

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(r.physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(r.physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            r.computeQueueFamily = i;
            break;
        }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = r.computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions;
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(r.physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> availableDevExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(r.physicalDevice, nullptr, &devExtCount, availableDevExts.data());

    for (const auto &ext : availableDevExts) {
        if (strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0) {
            deviceExtensions.push_back("VK_KHR_portability_subset");
            break;
        }
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(r.physicalDevice, &deviceCreateInfo, nullptr, &r.device) != VK_SUCCESS) {
        std::cerr << "[Vulkan] Failed to create logical device\n";
        return false;
    }

    vkGetDeviceQueue(r.device, r.computeQueueFamily, 0, &r.computeQueue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = r.computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(r.device, &poolInfo, nullptr, &r.commandPool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = r.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(r.device, &allocInfo, &r.commandBuffer) != VK_SUCCESS) return false;

    // Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        // Cascade level probe grids: binding (10 + level), one array layer per room surface.
        { 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(r.device, &layoutInfo, nullptr, &r.descSetLayout) != VK_SUCCESS) return false;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &r.descSetLayout;
    if (vkCreatePipelineLayout(r.device, &pipelineLayoutInfo, nullptr, &r.pipelineLayout) != VK_SUCCESS) return false;

    // Load SPIR-V Shaders
    std::string baseSpv = resolveExistingPath({
        "Vulkan/shaders/filter_atlas.spv",
        "shaders/filter_atlas.spv"
    });
    std::string baseDir = baseSpv.substr(0, baseSpv.find_last_of("/\\") + 1);

    r.cascadeGatherPipeline[0] = createComputePipeline(r.device, r.pipelineLayout, baseDir + "cascade_gather0.spv");
    r.cascadeGatherPipeline[1] = createComputePipeline(r.device, r.pipelineLayout, baseDir + "cascade_gather1.spv");
    r.cascadeGatherPipeline[2] = createComputePipeline(r.device, r.pipelineLayout, baseDir + "cascade_gather2.spv");
    r.cascadeGatherPipeline[3] = createComputePipeline(r.device, r.pipelineLayout, baseDir + "cascade_gather3.spv");
    r.cascadeIntegratePipeline = createComputePipeline(r.device, r.pipelineLayout, baseDir + "cascade_integrate.spv");
    r.filterAtlasPipeline      = createComputePipeline(r.device, r.pipelineLayout, baseDir + "filter_atlas.spv");
    r.causticsGenPipeline      = createComputePipeline(r.device, r.pipelineLayout, baseDir + "caustics_generate.spv");
    r.causticsFilterPipeline   = createComputePipeline(r.device, r.pipelineLayout, baseDir + "caustics_filter.spv");
    r.scenePipeline            = createComputePipeline(r.device, r.pipelineLayout, baseDir + "render_scene.spv");

    if (!r.cascadeGatherPipeline[0] || !r.cascadeGatherPipeline[1] || !r.cascadeGatherPipeline[2] || !r.cascadeGatherPipeline[3] ||
        !r.cascadeIntegratePipeline || !r.filterAtlasPipeline || !r.causticsGenPipeline || !r.causticsFilterPipeline || !r.scenePipeline) {
        std::cerr << "[Vulkan] Error creating one or more compute pipelines\n";
        return false;
    }

    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(r.device, &samplerInfo, nullptr, &r.linearSampler) != VK_SUCCESS) return false;

    // Images
    VkImageUsageFlags imgUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    createImage(r, 320, 64, VK_FORMAT_R32G32B32A32_SFLOAT, imgUsage, r.irradianceAtlas);
    createImage(r, 320, 64, VK_FORMAT_R32G32B32A32_SFLOAT, imgUsage, r.filteredAtlas);
    createImage(r, 1024, 1024, VK_FORMAT_R32G32B32A32_SFLOAT, imgUsage, r.causticTexture);
    createImage(r, r.width, r.height, VK_FORMAT_R32G32B32A32_SFLOAT, imgUsage, r.outTexture);

    // One array texture per cascade level (5 layers, one per room surface),
    // sized probesPerAxis x probesPerAxis probes with `rays` directions packed
    // per probe along X. Each is fully overwritten by its gather pass every
    // frame, so unlike the atlas images above they don't need a startup clear.
    struct CascadeLevelDims { uint32_t probesPerAxis; uint32_t rays; };
    static constexpr CascadeLevelDims kCascadeLevelDims[4] = {
        { 64, 16 }, { 32, 64 }, { 16, 256 }, { 8, 1024 },
    };
    VkImageUsageFlags cascadeUsage = VK_IMAGE_USAGE_STORAGE_BIT;
    for (int level = 0; level < 4; level++) {
        uint32_t w = kCascadeLevelDims[level].probesPerAxis * kCascadeLevelDims[level].rays;
        uint32_t h = kCascadeLevelDims[level].probesPerAxis;
        createImageArray(r, w, h, 5, VK_FORMAT_R32G32B32A32_SFLOAT, cascadeUsage, r.cascadeTex[level]);
    }

    // Initial Image Transitions
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(r.commandBuffer, &beginInfo);
    transitionImageLayout(r.commandBuffer, r.irradianceAtlas.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    transitionImageLayout(r.commandBuffer, r.filteredAtlas.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    transitionImageLayout(r.commandBuffer, r.causticTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    transitionImageLayout(r.commandBuffer, r.outTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    for (int level = 0; level < 4; level++) {
        transitionImageLayout(r.commandBuffer, r.cascadeTex[level].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    }
    vkEndCommandBuffer(r.commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &r.commandBuffer;
    vkQueueSubmit(r.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(r.computeQueue);

    // Buffers
    createBuffer(r, sizeof(GlassUniformsVK), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, r.uniformBuffer);

    TeapotMesh teapot;
    if (teapot.loadFromBinary(teapotBinPath)) {
        r.numTeapotNodes = static_cast<uint32_t>(teapot.nodes.size());
        r.numTeapotTris = static_cast<uint32_t>(teapot.triangles.size());

        createBuffer(r, teapot.nodes.size() * sizeof(GPUBVHNode), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, r.bvhBuffer);
        void *data;
        vkMapMemory(r.device, r.bvhBuffer.memory, 0, r.bvhBuffer.size, 0, &data);
        memcpy(data, teapot.nodes.data(), r.bvhBuffer.size);
        vkUnmapMemory(r.device, r.bvhBuffer.memory);

        createBuffer(r, teapot.triangles.size() * sizeof(GPUTriangle), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, r.triBuffer);
        vkMapMemory(r.device, r.triBuffer.memory, 0, r.triBuffer.size, 0, &data);
        memcpy(data, teapot.triangles.data(), r.triBuffer.size);
        vkUnmapMemory(r.device, r.triBuffer.memory);
    } else {
        createBuffer(r, 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, r.bvhBuffer);
        createBuffer(r, 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, r.triBuffer);
    }

    size_t causticSize = 1024 * 1024 * 4 * sizeof(uint32_t);
    createBuffer(r, causticSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, r.causticBuffer);

    VkDeviceSize stageSize = r.width * r.height * 4 * sizeof(float);
    createBuffer(r, stageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, r.stagingBuffer);

    // Descriptor Pool & Set
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 }
    };

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCreateInfo.pPoolSizes = poolSizes.data();
    poolCreateInfo.maxSets = 1;
    if (vkCreateDescriptorPool(r.device, &poolCreateInfo, nullptr, &r.descPool) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo setAllocInfo{};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = r.descPool;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts = &r.descSetLayout;
    if (vkAllocateDescriptorSets(r.device, &setAllocInfo, &r.descSet) != VK_SUCCESS) return false;

    VkDescriptorBufferInfo uboInfo{ r.uniformBuffer.buffer, 0, sizeof(GlassUniformsVK) };
    VkDescriptorBufferInfo bvhInfo{ r.bvhBuffer.buffer, 0, r.bvhBuffer.size };
    VkDescriptorBufferInfo triInfo{ r.triBuffer.buffer, 0, r.triBuffer.size };
    VkDescriptorBufferInfo cstInfo{ r.causticBuffer.buffer, 0, r.causticBuffer.size };

    VkDescriptorImageInfo imgAtlasInfo{ VK_NULL_HANDLE, r.irradianceAtlas.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo imgFiltInfo{ VK_NULL_HANDLE, r.filteredAtlas.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo imgCausticInfo{ VK_NULL_HANDLE, r.causticTexture.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo imgOutInfo{ VK_NULL_HANDLE, r.outTexture.view, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorImageInfo smpAtlasInfo{ r.linearSampler, r.filteredAtlas.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo smpCausticInfo{ r.linearSampler, r.causticTexture.view, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorImageInfo imgCascadeInfo[4];
    for (int level = 0; level < 4; level++) {
        imgCascadeInfo[level] = { VK_NULL_HANDLE, r.cascadeTex[level].view, VK_IMAGE_LAYOUT_GENERAL };
    }

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bvhInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &cstInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgAtlasInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgFiltInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgCausticInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgOutInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &smpAtlasInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &smpCausticInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgCascadeInfo[0], nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgCascadeInfo[1], nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgCascadeInfo[2], nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, r.descSet, 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imgCascadeInfo[3], nullptr, nullptr }
    };

    vkUpdateDescriptorSets(r.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    return true;
}

void renderFrameVK(VulkanRenderer &r, float deltaTime) {
    if (!r.sunPaused) {
        r.sunTime += deltaTime * r.sunSpeed;
    }

    float sunAzimuth = r.sunTime * 0.35f + r.manualAzimuthOffset;
    float sunX = -0.78f + std::sin(sunAzimuth) * 0.12f;
    float sunY =  0.55f + std::cos(sunAzimuth * 0.7f) * 0.10f + r.manualElevationOffset;
    float sunZ =  std::cos(sunAzimuth) * 0.42f + 0.05f;
    glm::vec3 sunDir = glm::normalize(glm::vec3(sunX, sunY, sunZ));

    GlassUniformsVK uniforms;
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
    uniforms.numTeapotTris = r.numTeapotTris;
    uniforms.pad = glm::vec2(0.0f);

    void *data;
    vkMapMemory(r.device, r.uniformBuffer.memory, 0, sizeof(GlassUniformsVK), 0, &data);
    memcpy(data, &uniforms, sizeof(GlassUniformsVK));
    vkUnmapMemory(r.device, r.uniformBuffer.memory);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(r.commandBuffer, &beginInfo);

    vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);

    VkMemoryBarrier b1{};
    b1.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    if (r.renderMode != 0) {
        // Far-to-near: level 3 has nothing above it to read, level 0 reads
        // level 1, and so on down. Each dispatch covers exactly that level's
        // probe x ray grid, with one z-slice per room surface.
        struct CascadeLevelDims { uint32_t probesPerAxis; uint32_t rays; };
        static constexpr CascadeLevelDims kCascadeLevelDims[4] = {
            { 64, 16 }, { 32, 64 }, { 16, 256 }, { 8, 1024 },
        };
        for (int level = 3; level >= 0; level--) {
            uint32_t levelWidth = kCascadeLevelDims[level].probesPerAxis * kCascadeLevelDims[level].rays;
            uint32_t probesPerAxis = kCascadeLevelDims[level].probesPerAxis;

            vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);
            vkCmdBindPipeline(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.cascadeGatherPipeline[level]);
            vkCmdDispatch(r.commandBuffer, (levelWidth + 15) / 16, (probesPerAxis + 15) / 16, 5);
            vkCmdPipelineBarrier(r.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b1, 0, nullptr, 0, nullptr);
        }

        vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);
        vkCmdBindPipeline(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.cascadeIntegratePipeline);
        vkCmdDispatch(r.commandBuffer, (320 + 15) / 16, (64 + 15) / 16, 1);
        vkCmdPipelineBarrier(r.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b1, 0, nullptr, 0, nullptr);

        vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);
        vkCmdBindPipeline(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.filterAtlasPipeline);
        vkCmdDispatch(r.commandBuffer, (320 + 15) / 16, (64 + 15) / 16, 1);

        vkCmdPipelineBarrier(r.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b1, 0, nullptr, 0, nullptr);

        vkCmdFillBuffer(r.commandBuffer, r.causticBuffer.buffer, 0, r.causticBuffer.size, 0);

        VkMemoryBarrier bClear{};
        bClear.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        bClear.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bClear.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(r.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bClear, 0, nullptr, 0, nullptr);

        vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);
        vkCmdBindPipeline(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.causticsGenPipeline);
        vkCmdDispatch(r.commandBuffer, 2048 / 16, 2048 / 16, 1);

        VkMemoryBarrier b2{};
        b2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(r.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b2, 0, nullptr, 0, nullptr);

        vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);
        vkCmdBindPipeline(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.causticsFilterPipeline);
        vkCmdDispatch(r.commandBuffer, 1024 / 16, 1024 / 16, 1);

        vkCmdPipelineBarrier(r.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b2, 0, nullptr, 0, nullptr);
    }

    vkCmdBindDescriptorSets(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipelineLayout, 0, 1, &r.descSet, 0, nullptr);
    vkCmdBindPipeline(r.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.scenePipeline);
    vkCmdDispatch(r.commandBuffer, (r.width + 15) / 16, (r.height + 15) / 16, 1);

    vkEndCommandBuffer(r.commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &r.commandBuffer;

    VkResult subRes = vkQueueSubmit(r.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (subRes != VK_SUCCESS) {
        std::cerr << "[Vulkan] vkQueueSubmit failed: " << subRes << "\n";
    }
    VkResult waitRes = vkQueueWaitIdle(r.computeQueue);
    if (waitRes != VK_SUCCESS) {
        std::cerr << "[Vulkan] vkQueueWaitIdle failed: " << waitRes << "\n";
    }
}

static void readbackImage(VulkanRenderer &r, std::vector<float> &outPixels) {
    outPixels.resize(r.width * r.height * 4);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(r.commandBuffer, &beginInfo);

    transitionImageLayout(r.commandBuffer, r.outTexture.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = { 0, 0, 0 };
    copyRegion.imageExtent = { r.width, r.height, 1 };

    vkCmdCopyImageToBuffer(r.commandBuffer, r.outTexture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.stagingBuffer.buffer, 1, &copyRegion);

    transitionImageLayout(r.commandBuffer, r.outTexture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

    vkEndCommandBuffer(r.commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &r.commandBuffer;

    vkQueueSubmit(r.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(r.computeQueue);

    void *data;
    vkMapMemory(r.device, r.stagingBuffer.memory, 0, r.stagingBuffer.size, 0, &data);
    memcpy(outPixels.data(), data, r.stagingBuffer.size);
    vkUnmapMemory(r.device, r.stagingBuffer.memory);
}

int runHeadlessVK(VulkanRenderer &r) {
    std::cout << "\nVulkan, " << r.width << "x" << r.height
              << ", 20 frames per mode after 3 warm-up frames\n";

    system("mkdir -p output");

    struct ModeTest {
        uint32_t mode;
        std::string name;
        std::string filename;
    };

    std::vector<ModeTest> modes = {
        { 1, "clear glass", "rc_glass_vk_scene.png" },
        { 2, "frosted glass", "rc_glass_vk_frosted.png" },
        { 3, "high dispersion", "rc_glass_vk_dispersion.png" },
        { 0, "whitted baseline", "rc_glass_vk_whitted.png" }
    };

    for (const auto &test : modes) {
        r.renderMode = test.mode;

        for (int f = 0; f < 3; f++) {
            renderFrameVK(r, 0.016f);
        }

        const int numFrames = 20;
        auto tStart = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < numFrames; f++) {
            renderFrameVK(r, 0.016f);
        }
        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        double avgFrameMs = totalMs / double(numFrames);
        double fps = 1000.0 / avgFrameMs;

        std::vector<float> pixels;
        readbackImage(r, pixels);

        std::string outPath = "output/" + test.filename;
        saveFloatBufferToPNG(pixels.data(), r.width, r.height, outPath);

        std::cout << "  mode " << test.mode << "  " << test.name
                  << "  " << avgFrameMs << " ms (" << fps << " fps)"
                  << "  -> " << outPath << "\n";
    }

    return 0;
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

    if (!initVulkan(gVK, teapotBinPath)) {
        return 1;
    }

    if (headless) {
        return runHeadlessVK(gVK);
    }

    // If running in GUI mode, we also initialize GLFW window and run interactive loop
    if (!glfwInit()) {
        std::cerr << "[Vulkan] Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *win = glfwCreateWindow(gVK.width, gVK.height, "Radiance Cascades Glass & Caustics [Vulkan 1.2]", nullptr, nullptr);
    if (!win) {
        std::cerr << "[Vulkan] Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    std::cout << "Radiance Cascades Glass & Caustics [Vulkan 1.2]\n\n";
    std::cout << "  Controls:\n";
    std::cout << "    [Space]             : Toggle Sun Animation\n";
    std::cout << "    [1]                 : Clear Glass Mode\n";
    std::cout << "    [2]                 : Frosted Glass Mode\n";
    std::cout << "    [3]                 : High Dispersion Prism Mode\n";
    std::cout << "    [0]                 : Whitted Ray Tracing Baseline\n";
    std::cout << "    [+/-]               : Adjust Roughness\n";
    std::cout << "    [S]                 : Screenshot\n";
    std::cout << "    [ESC]               : Quit\n\n";

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) gVK.renderMode = 1;
        if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) gVK.renderMode = 2;
        if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) gVK.renderMode = 3;
        if (glfwGetKey(win, GLFW_KEY_0) == GLFW_PRESS) gVK.renderMode = 0;
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS) gVK.sunPaused = !gVK.sunPaused;

        double now = glfwGetTime();
        double dt = now - lastTime;
        lastTime = now;

        renderFrameVK(gVK, static_cast<float>(dt));

        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
            std::vector<float> px;
            readbackImage(gVK, px);
            system("mkdir -p output");
            saveFloatBufferToPNG(px.data(), gVK.width, gVK.height, "output/rc_glass_vk_snapshot.png");
            std::cout << "[Capture] Saved output/rc_glass_vk_snapshot.png\n";
        }
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
