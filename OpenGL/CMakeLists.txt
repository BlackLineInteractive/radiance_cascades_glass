cmake_minimum_required(VERSION 3.15)
project(rc_glass_opengl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../assets
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(rc_glass_gl
    src/main.cpp
    src/gl_loader.cpp
)

target_link_libraries(rc_glass_gl
    PRIVATE
    OpenGL::GL
    glfw
    glm::glm
)
