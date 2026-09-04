cmake_minimum_required(VERSION 3.15)
project(rc_glass_vulkan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Compile shaders
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADERS
    radiance_cascades.comp
    filter_atlas.comp
    caustics_generate.comp
    caustics_filter.comp
    render_scene.comp
)

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${GLSLANG_VALIDATOR} -V "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}" -o ${SPV_FILE}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
        COMMENT "Compiling SPIR-V shader: ${SHADER}"
    )
    list(APPEND SPV_FILES ${SPV_FILE})
endforeach()

add_custom_target(vulkan_shaders ALL DEPENDS ${SPV_FILES})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../assets
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
)

add_executable(rc_glass_vk
    src/main.cpp
)

add_dependencies(rc_glass_vk vulkan_shaders)

target_link_libraries(rc_glass_vk
    PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)
