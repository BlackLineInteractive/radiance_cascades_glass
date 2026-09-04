cmake_minimum_required(VERSION 3.19)
project(radiance_cascades_glass LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_METAL  "Build Apple Metal backend" ON)
option(BUILD_VULKAN "Build Vulkan 1.2+ backend" ON)
option(BUILD_OPENGL "Build OpenGL 4.3+ Core backend" ON)

# Vulkan Backend
if(BUILD_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "[Radiance Cascades] Vulkan found - adding Vulkan target")
        add_subdirectory(Vulkan)
    else()
        message(STATUS "[Radiance Cascades] Vulkan not found - skipping Vulkan target")
    endif()
endif()

# OpenGL Backend
if(BUILD_OPENGL)
    find_package(OpenGL QUIET)
    find_package(glfw3 QUIET)
    if(OpenGL_FOUND AND glfw3_FOUND)
        message(STATUS "[Radiance Cascades] OpenGL 4.3+ & GLFW found - adding OpenGL target")
        add_subdirectory(OpenGL)
    else()
        message(STATUS "[Radiance Cascades] OpenGL or GLFW not found - skipping OpenGL target")
    endif()
endif()

# Metal Backend (Apple platforms)
if(BUILD_METAL AND APPLE)
    message(STATUS "[Radiance Cascades] Apple Darwin detected - adding Metal target")

    find_program(XCRUN_BIN xcrun REQUIRED)

    set(METAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/Metal/shaders/RCGlassShaders.metal")
    set(METALLIB_OUT "${CMAKE_CURRENT_SOURCE_DIR}/Metal/shaders/RCGlassShaders.metallib")

    add_custom_command(
        OUTPUT ${METALLIB_OUT}
        COMMAND ${XCRUN_BIN} -sdk macosx metal -O3 -c ${METAL_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/RCGlassShaders.air
        COMMAND ${XCRUN_BIN} -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/RCGlassShaders.air -o ${METALLIB_OUT}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/RCGlassShaders.air
        DEPENDS ${METAL_SRC}
        COMMENT "Compiling Metal shaders to metallib"
    )

    add_custom_target(metal_shaders ALL DEPENDS ${METALLIB_OUT})

    add_executable(rc_glass_app
        Metal/src/main.mm
    )
    add_dependencies(rc_glass_app metal_shaders)

    target_include_directories(rc_glass_app PRIVATE
        assets
        common
        Metal/src
    )

    target_compile_options(rc_glass_app PRIVATE -fobjc-arc -O3 -Wall)

    target_link_libraries(rc_glass_app PRIVATE
        "-framework Cocoa"
        "-framework Metal"
        "-framework MetalKit"
        "-framework QuartzCore"
    )
endif()
