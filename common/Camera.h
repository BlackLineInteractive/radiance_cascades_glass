#pragma once

#include <cmath>
#include <algorithm>

#if defined(USE_GLM) || !defined(__APPLE__)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class OrbitCamera {
public:
    glm::vec3 target;
    float distance;
    float yaw;
    float pitch;
    float fovY;
    float aspect;

    OrbitCamera()
        : target(0.0f, 0.38f, 0.15f),
          distance(2.9f),
          yaw(-0.32f),
          pitch(0.34f),
          fovY(48.0f * 3.14159265f / 180.0f),
          aspect(16.0f / 9.0f) {}

    void orbit(float deltaX, float deltaY) {
        yaw += deltaX * 0.006f;
        pitch += deltaY * 0.006f;
        const float maxPitch = 1.45f;
        const float minPitch = -0.15f;
        pitch = std::max(minPitch, std::min(maxPitch, pitch));
    }

    void zoom(float delta) {
        distance -= delta * 0.25f;
        distance = std::max(1.0f, std::min(8.0f, distance));
    }

    void pan(float deltaX, float deltaY) {
        glm::vec3 eye = getPosition();
        glm::vec3 forward = glm::normalize(target - eye);
        glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), forward));
        glm::vec3 up = glm::cross(forward, right);

        target += (right * deltaX + up * deltaY) * (distance * 0.0015f);
        target.y = std::max(0.1f, std::min(2.5f, target.y));
    }

    glm::vec3 getPosition() const {
        float x = target.x + distance * std::cos(pitch) * std::sin(yaw);
        float y = target.y + distance * std::sin(pitch);
        float z = target.z - distance * std::cos(pitch) * std::cos(yaw);
        return glm::vec3(x, y, z);
    }

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(getPosition(), target, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 getProjectionMatrix(bool reverseZ = false) const {
        glm::mat4 p = glm::perspective(fovY, aspect, 0.1f, 100.0f);
        if (reverseZ) {
            p[2][2] = 0.0f;
            p[3][2] = 0.1f;
        }
        return p;
    }
};

#else

#include <simd/simd.h>

class OrbitCamera {
public:
    simd::float3 target;
    float distance;
    float yaw;
    float pitch;
    float fovY;
    float aspect;

    OrbitCamera()
        : target(simd::make_float3(0.0f, 0.38f, 0.15f)),
          distance(2.9f),
          yaw(-0.32f),
          pitch(0.34f),
          fovY(48.0f * 3.14159265f / 180.0f),
          aspect(16.0f / 9.0f) {}

    void orbit(float deltaX, float deltaY) {
        yaw += deltaX * 0.006f;
        pitch += deltaY * 0.006f;
        const float maxPitch = 1.45f;
        const float minPitch = -0.15f;
        pitch = std::max(minPitch, std::min(maxPitch, pitch));
    }

    void zoom(float delta) {
        distance -= delta * 0.25f;
        distance = std::max(1.0f, std::min(8.0f, distance));
    }

    void pan(float deltaX, float deltaY) {
        simd::float3 eye = getPosition();
        simd::float3 forward = simd::normalize(target - eye);
        simd::float3 right = simd::normalize(simd::cross(simd::make_float3(0.0f, 1.0f, 0.0f), forward));
        simd::float3 up = simd::cross(forward, right);

        target += (right * deltaX + up * deltaY) * (distance * 0.0015f);
        target.y = std::max(0.1f, std::min(2.5f, target.y));
    }

    simd::float3 getPosition() const {
        float x = target.x + distance * std::cos(pitch) * std::sin(yaw);
        float y = target.y + distance * std::sin(pitch);
        float z = target.z - distance * std::cos(pitch) * std::cos(yaw);
        return simd::make_float3(x, y, z);
    }

    simd::float4x4 getViewMatrix() const {
        simd::float3 eye = getPosition();
        simd::float3 f = simd::normalize(target - eye);
        simd::float3 s = simd::normalize(simd::cross(f, simd::make_float3(0.0f, 1.0f, 0.0f)));
        simd::float3 u = simd::cross(s, f);

        simd::float4x4 m;
        m.columns[0] = simd::make_float4(s.x, u.x, -f.x, 0.0f);
        m.columns[1] = simd::make_float4(s.y, u.y, -f.y, 0.0f);
        m.columns[2] = simd::make_float4(s.z, u.z, -f.z, 0.0f);
        m.columns[3] = simd::make_float4(-simd::dot(s, eye), -simd::dot(u, eye), simd::dot(f, eye), 1.0f);
        return m;
    }

    simd::float4x4 getProjectionMatrix() const {
        float tanHalfFov = std::tan(fovY * 0.5f);
        float nearZ = 0.1f;
        float farZ = 100.0f;

        simd::float4x4 m = {};
        m.columns[0].x = 1.0f / (aspect * tanHalfFov);
        m.columns[1].y = 1.0f / tanHalfFov;
        m.columns[2].z = -(farZ + nearZ) / (farZ - nearZ);
        m.columns[2].w = -1.0f;
        m.columns[3].z = -(2.0f * farZ * nearZ) / (farZ - nearZ);
        return m;
    }
};

#endif
