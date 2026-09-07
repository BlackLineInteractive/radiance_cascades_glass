void main() {
    uvec2 tid = gl_GlobalInvocationID.xy;
    if (tid.x >= kAtlasSurfaceWidth * kNumSurfaces || tid.y >= kAtlasSurfaceHeight) return;

    uint surfaceId = tid.x / kAtlasSurfaceWidth;
    uint probeX = tid.x % kAtlasSurfaceWidth;
    uint probeY = tid.y;

    uint base = probeX * CASCADE_RAYS_THIS;

    vec3 sum = vec3(0.0);
    for (uint i = 0u; i < CASCADE_RAYS_THIS; i++) {
        sum += imageLoad(cascade0Tex, ivec3(int(base + i), int(probeY), int(surfaceId))).rgb;
    }
    vec3 newIrradiance = sum / float(CASCADE_RAYS_THIS);

    if (uniforms.frameIndex > 1u) {
        newIrradiance = mix(newIrradiance, imageLoad(irradianceAtlas, ivec2(tid)).rgb, 0.70);
    }
    imageStore(irradianceAtlas, ivec2(tid), vec4(newIrradiance, 1.0));
}
