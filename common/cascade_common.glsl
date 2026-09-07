vec3 cascadeDirection(Basis tbn, int index, int count, float jitter) {
    float cosTheta = sqrt(max(0.0, 1.0 - (float(index) + 0.5) / float(count)));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = float(index) * 2.399963229728 + jitter;
    return basisToWorld(tbn, vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));
}

void cascadeProbeAt(uint surfaceId, uint probeX, uint probeY, uint probesPerAxis,
                    out vec3 origin, out Basis tbn, out vec2 uv) {
    uv = (vec2(probeX, probeY) + 0.5) / float(probesPerAxis);
    vec3 pos, nor;
    getSurfaceGeometry(surfaceId, uv, pos, nor);
    tbn = makeTBN(nor);
    origin = pos + nor * 0.004;
}

vec3 evalSurfaceDirectLighting(HitRecord hit, vec3 sunDir, vec3 sunCol, float sunInt, uint numNodes) {
    float NdotL = max(0.0, dot(hit.normal, sunDir));
    vec3 sunIllum = vec3(0.0);
    if (NdotL > 0.0) {
        Ray sRay;
        sRay.origin = hit.position + hit.normal * 0.002;
        sRay.direction = sunDir;
        HitRecord sHit = intersectSceneInterval(sRay, 0.001, 100.0, false, numNodes);
        if (!sHit.hit) {
            sunIllum = sunCol * (sunInt * NdotL);
        }
    }
    vec3 ambient = vec3(0.08);
    return (sunIllum + ambient) * hit.albedo;
}

vec4 traceCascadeInterval(vec3 origin, vec3 dir, float tMin, float tMax, vec3 sunDir, vec3 sunCol, float sunInt, uint numNodes) {
    Ray probeRay;
    probeRay.origin = origin;
    probeRay.direction = dir;
    HitRecord hit = intersectSceneInterval(probeRay, tMin, tMax, false, numNodes);
    if (hit.hit) {
        vec3 hitRad = evalSurfaceDirectLighting(hit, sunDir, sunCol, sunInt, numNodes);

        float tNorm = clamp((hit.distance - tMin) / max(1e-4, tMax - tMin), 0.0, 1.0);
        float boundaryFade = smoothstep(0.85, 1.0, tNorm);
        return vec4(hitRad, boundaryFade);
    }
    return vec4(0.0, 0.0, 0.0, 1.0);
}

#define CASCADE_BILINEAR(cascadeImg, resultVar, surfaceIdVal, uvVal, probesVal, raysVal, dirVal) do { \
    float _gx = (uvVal).x * float(probesVal) - 0.5; \
    float _gy = (uvVal).y * float(probesVal) - 0.5; \
    int _x0 = int(floor(_gx)); \
    int _y0 = int(floor(_gy)); \
    float _fx = _gx - float(_x0); \
    float _fy = _gy - float(_y0); \
    int _maxIdx = int(probesVal) - 1; \
    int _x1 = clamp(_x0 + 1, 0, _maxIdx); \
    int _y1 = clamp(_y0 + 1, 0, _maxIdx); \
    _x0 = clamp(_x0, 0, _maxIdx); \
    _y0 = clamp(_y0, 0, _maxIdx); \
    int _layer = int(surfaceIdVal); \
    int _d = int(dirVal); \
    int _r = int(raysVal); \
    vec3 _c00 = imageLoad(cascadeImg, ivec3(_x0 * _r + _d, _y0, _layer)).rgb; \
    vec3 _c10 = imageLoad(cascadeImg, ivec3(_x1 * _r + _d, _y0, _layer)).rgb; \
    vec3 _c01 = imageLoad(cascadeImg, ivec3(_x0 * _r + _d, _y1, _layer)).rgb; \
    vec3 _c11 = imageLoad(cascadeImg, ivec3(_x1 * _r + _d, _y1, _layer)).rgb; \
    resultVar = mix(mix(_c00, _c10, _fx), mix(_c01, _c11, _fx), _fy); \
} while (false)
