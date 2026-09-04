// One invocation per (probe, direction) at this cascade level, dispatched once
// per level, far-to-near. Each level traces its own probe grid exactly once
// and reads the level above through sampleCascadeBilinear rather than
// retracing it, so a coarse cascade costs what its own probe count and ray
// count say it costs, not what the finest level below it costs.
void main() {
    uvec3 tid = gl_GlobalInvocationID;
    uint perSurfaceWidth = CASCADE_PROBES_THIS * CASCADE_RAYS_THIS;
    if (tid.x >= perSurfaceWidth || tid.y >= CASCADE_PROBES_THIS || tid.z >= kNumSurfaces) return;

    uint surfaceId = tid.z;
    uint probeX = tid.x / CASCADE_RAYS_THIS;
    uint dirIndex = tid.x % CASCADE_RAYS_THIS;
    uint probeY = tid.y;

    vec3 origin;
    Basis tbn;
    vec2 uv;
    cascadeProbeAt(surfaceId, probeX, probeY, CASCADE_PROBES_THIS, origin, tbn, uv);

    // Rotates a little every frame so the temporal blend in the integrate
    // pass averages away noise instead of freezing it.
    float jitterSeed = float(surfaceId) * 37.0 + float(uniforms.frameIndex) * 0.6180339887;
    float jitter = fract(sin(dot(vec2(probeX, probeY) + jitterSeed, vec2(12.9898, 78.233))) * 43758.5453) * (2.0 * kPi);

    vec3 sunDir = normalize(uniforms.sunDirection);
    vec3 dir = cascadeDirection(tbn, int(dirIndex), int(CASCADE_RAYS_THIS), jitter);
    vec4 seg = traceCascadeInterval(origin, dir, CASCADE_TMIN, CASCADE_TMAX,
                                    sunDir, uniforms.sunColor, uniforms.sunIntensity, uniforms.numTeapotNodes);

    vec3 result;
#ifdef CASCADE_HAS_UPPER
    if (seg.w <= 0.0) {
        result = seg.rgb;
    } else {
        vec3 fromUpper = vec3(0.0);
        uint upperBase = dirIndex * 4u;
        for (uint k = 0u; k < 4u; k++) {
            vec3 upperSample;
            CASCADE_BILINEAR(upperCascade, upperSample, surfaceId, uv,
                             CASCADE_PROBES_UPPER, CASCADE_RAYS_UPPER, upperBase + k);
            fromUpper += 0.25 * upperSample;
        }
        result = seg.rgb + seg.w * fromUpper;
    }
#else
    result = seg.w > 0.0 ? seg.rgb + seg.w * getSkyRadiance(dir, sunDir) * CASCADE_SKY_BOOST : seg.rgb;
#endif

    imageStore(outCascade, ivec3(tid.x, tid.y, surfaceId), vec4(result, 1.0));
}
