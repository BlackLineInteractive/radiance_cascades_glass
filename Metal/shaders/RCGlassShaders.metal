#include <metal_stdlib>
using namespace metal;

constant float kPi = 3.14159265358979323846f;
constant uint kCausticRes = 1024;
constant float kFloorMinX = -2.5f;
constant float kFloorMaxX =  2.5f;
constant float kFloorMinZ = -2.5f;
constant float kFloorMaxZ =  2.5f;

constant float3 kSphereCenter = float3(1.10f, 0.45f, -0.25f);
constant float  kSphereRadius = 0.45f;

constant float3 kCylinderCenter = float3(1.35f, 0.06f, 0.70f);
constant float  kCylinderRadius = 0.28f;
constant float  kCylinderHeight = 0.80f;

constant float3 kPrismCenter = float3(-1.25f, 0.0f, -0.20f);
constant float  kPrismSide   = 0.55f;
constant float  kPrismHeight = 0.70f;

constant float kRoomMinX = -2.5f;
constant float kRoomMaxX =  2.5f;
constant float kRoomMinY =  0.0f;
constant float kRoomMaxY =  3.5f;
constant float kRoomMinZ = -2.5f;
constant float kRoomMaxZ =  2.5f;

constant float kWinMinY = 0.35f;
constant float kWinMaxY = 3.25f;
constant float kWinMinZ = -2.10f;
constant float kWinMaxZ =  2.10f;

struct Ray {
    float3 origin;
    float3 direction;
};

struct HitRecord {
    bool hit;
    float distance;
    float3 position;
    float3 normal;
    float3 albedo;
    float roughness;
    bool isGlass;
    uint objectId;
    float ior;
    float dispersion;
    float3 absorption;
};

struct GPUBVHNode {
    float4 bmin;
    float4 bmax;
    int leftChild;
    int rightChild;
    int pad[2];
};

struct GPUTriangle {
    float4 v0;
    float4 v1;
    float4 v2;
    float4 n0;
    float4 n1;
    float4 n2;
};

struct GlassUniforms {
    float4x4 viewInverse;
    float4x4 projectionInverse;
    float3 cameraPosition;
    float time;

    float3 sunDirection;
    float sunIntensity;
    float3 sunColor;
    float ambientIntensity;

    float glassIor;
    float glassDispersion;
    float glassRoughness;
    float glassAbsorption;

    uint renderMode;
    uint width;
    uint height;
    uint frameIndex;

    uint numTeapotNodes;
    uint numTeapotTris;
    uint ablationMask;
    uint pad;
};

// Ablation switches. Every bit set = the full technique; clearing one turns a
// single contribution off so its cost and its error can be attributed.
// Bit 2 (atlas filter) and bit 4 (cascade merge) are honoured on the host - they
// decide which atlas is bound and how many cascade levels get dispatched - so
// only the bits the shader itself reads are named here.
constant uint kAblCascadeGI    = 1u << 0;
constant uint kAblCaustics     = 1u << 1;
constant uint kAblTemporal     = 1u << 3;
constant uint kAblDispersion   = 1u << 5;

// Path-traced reference controls. Kept in their own buffer so the reference
// pass can be re-parameterised without touching the real-time uniforms.
struct PathTraceParams {
    uint samplesPerLaunch;   // paths traced per pixel by this dispatch
    uint sampleBase;         // paths already accumulated (also the RNG decorrelator)
    uint maxDepth;           // bounces before the path is cut
    uint rrStartDepth;       // Russian roulette kicks in at this depth
    float sunAngularRadius;  // radians; 0.00465 is the real sun, wider converges faster
    float indirectClamp;     // firefly clamp on a single path contribution, <=0 disables
    float exposure;          // shared with the raster path so both tone map identically
    uint seedOffset;         // decorrelates an equal-time render from the reference
};

inline float dielectricFresnel(float cosThetaI, float iorI, float iorT) {
    cosThetaI = clamp(abs(cosThetaI), 0.0f, 1.0f);
    float sinThetaI = sqrt(max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = (iorI / iorT) * sinThetaI;
    if (sinThetaT >= 1.0f) return 1.0f;

    float cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rParallel = ((iorT * cosThetaI) - (iorI * cosThetaT)) /
                      ((iorT * cosThetaI) + (iorI * cosThetaT));
    float rPerp     = ((iorI * cosThetaI) - (iorT * cosThetaT)) /
                      ((iorI * cosThetaI) + (iorT * cosThetaT));
    return 0.5f * (rParallel * rParallel + rPerp * rPerp);
}

inline bool refractRay(float3 I, float3 N, float eta, thread float3 &T) {
    if (dot(N, I) > 0.0f) N = -N;
    float cosI = -dot(N, I);
    float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T >= 1.0f) return false;
    float cosT = sqrt(1.0f - sin2T);
    T = eta * I + (eta * cosI - cosT) * N;
    return true;
}

inline float3 beerLambertAbsorption(float3 absorptionCoeff, float distance) {
    return exp(-absorptionCoeff * distance);
}

inline bool intersectBox(Ray ray, float3 bmin, float3 bmax, float tMin, thread float &tHit, thread float3 &hitNormal) {
    float3 invD = 1.0f / (ray.direction + float3(1e-12f));
    float3 t0 = (bmin - ray.origin) * invD;
    float3 t1 = (bmax - ray.origin) * invD;

    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);

    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);

    if (enter > exit || exit < tMin) return false;

    float t = enter > tMin ? enter : exit;
    tHit = t;

    float3 p = ray.origin + ray.direction * t;
    float3 center = 0.5f * (bmin + bmax);
    float3 d = p - center;
    float3 extent = 0.5f * (bmax - bmin);

    float3 bias = d / extent;
    float3 absBias = abs(bias);

    if (absBias.x > absBias.y && absBias.x > absBias.z) {
        hitNormal = float3(sign(bias.x), 0.0f, 0.0f);
    } else if (absBias.y > absBias.z) {
        hitNormal = float3(0.0f, sign(bias.y), 0.0f);
    } else {
        hitNormal = float3(0.0f, 0.0f, sign(bias.z));
    }
    return true;
}

inline bool intersectBoxFast(Ray ray, float3 bmin, float3 bmax, thread float &tNear) {
    float3 invD = 1.0f / (ray.direction + float3(1e-12f));
    float3 t0 = (bmin - ray.origin) * invD;
    float3 t1 = (bmax - ray.origin) * invD;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);
    tNear = enter;
    return (enter <= exit && exit > 0.001f);
}

inline bool intersectSphere(Ray ray, float3 center, float radius, float tMin, thread float &tHit, thread float3 &hitNormal) {
    float3 oc = ray.origin - center;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return false;

    float sqrtDisc = sqrt(disc);
    float t = -b - sqrtDisc;
    if (t < tMin) t = -b + sqrtDisc;
    if (t < tMin) return false;

    tHit = t;
    hitNormal = normalize((ray.origin + ray.direction * t) - center);
    return true;
}

inline bool intersectCylinder(Ray ray, float3 base, float radius, float height, float tMin, thread float &tHit, thread float3 &hitNormal) {
    float3 d = ray.direction;
    float3 o = ray.origin - base;

    float a = d.x * d.x + d.z * d.z;
    float b = 2.0f * (o.x * d.x + o.z * d.z);
    float c = o.x * o.x + o.z * o.z - radius * radius;

    float tClosest = 1e30f;
    float3 bestNorm = float3(0.0f);
    bool found = false;

    if (a > 1e-6f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sqrtD = sqrt(disc);
            float t0 = (-b - sqrtD) / (2.0f * a);
            float t1 = (-b + sqrtD) / (2.0f * a);

            if (t0 > tMin) {
                float y = o.y + d.y * t0;
                if (y >= 0.0f && y <= height) {
                    tClosest = t0;
                    bestNorm = normalize(float3(o.x + d.x * t0, 0.0f, o.z + d.z * t0));
                    found = true;
                }
            }
            if (!found && t1 > tMin) {
                float y = o.y + d.y * t1;
                if (y >= 0.0f && y <= height) {
                    tClosest = t1;
                    bestNorm = normalize(float3(o.x + d.x * t1, 0.0f, o.z + d.z * t1));
                    found = true;
                }
            }
        }
    }

    if (abs(d.y) > 1e-6f) {
        float tCap = (height - o.y) / d.y;
        if (tCap > tMin && tCap < tClosest) {
            float x = o.x + d.x * tCap;
            float z = o.z + d.z * tCap;
            if (x * x + z * z <= radius * radius) {
                tClosest = tCap;
                bestNorm = float3(0.0f, 1.0f, 0.0f);
                found = true;
            }
        }
        float tBase = -o.y / d.y;
        if (tBase > tMin && tBase < tClosest) {
            float x = o.x + d.x * tBase;
            float z = o.z + d.z * tBase;
            if (x * x + z * z <= radius * radius) {
                tClosest = tBase;
                bestNorm = float3(0.0f, -1.0f, 0.0f);
                found = true;
            }
        }
    }

    if (found) {
        tHit = tClosest;
        hitNormal = bestNorm;
        return true;
    }
    return false;
}

inline bool intersectTriangularPrism(Ray ray, float3 baseCenter, float side, float height, float tMin, thread float &tHit, thread float3 &hitNormal) {
    float h = side * 0.8660254f;
    float3 p0 = baseCenter + float3(0.0f, 0.0f, 2.0f * h / 3.0f);
    float3 p1 = baseCenter + float3(-side * 0.5f, 0.0f, -h / 3.0f);
    float3 p2 = baseCenter + float3( side * 0.5f, 0.0f, -h / 3.0f);

    float tClosest = 1e30f;
    float3 bestNorm = float3(0.0f);
    bool found = false;

    float3 pts[3] = { p0, p1, p2 };
    for (int i = 0; i < 3; i++) {
        float3 a = pts[i];
        float3 b = pts[(i + 1) % 3];
        float3 edge = b - a;
        float3 sideNorm = normalize(float3(edge.z, 0.0f, -edge.x));

        float denom = dot(ray.direction, sideNorm);
        if (abs(denom) > 1e-6f) {
            float t = dot(a - ray.origin, sideNorm) / denom;
            if (t > tMin && t < tClosest) {
                float3 p = ray.origin + ray.direction * t;
                if (p.y >= baseCenter.y && p.y <= baseCenter.y + height) {
                    float3 ap = p - a;
                    float edgeLen = length(edge);
                    float proj = dot(ap, edge) / (edgeLen * edgeLen);
                    if (proj >= 0.0f && proj <= 1.0f) {
                        tClosest = t;
                        bestNorm = sideNorm;
                        found = true;
                    }
                }
            }
        }
    }

    if (abs(ray.direction.y) > 1e-6f) {
        float tTop = (baseCenter.y + height - ray.origin.y) / ray.direction.y;
        if (tTop > tMin && tTop < tClosest) {
            float3 p = ray.origin + ray.direction * tTop;
            float2 v0 = p2.xz - p0.xz;
            float2 v1 = p1.xz - p0.xz;
            float2 v2 = p.xz - p0.xz;
            float dot00 = dot(v0, v0);
            float dot01 = dot(v0, v1);
            float dot02 = dot(v0, v2);
            float dot11 = dot(v1, v1);
            float dot12 = dot(v1, v2);
            float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
            if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
                tClosest = tTop;
                bestNorm = float3(0.0f, 1.0f, 0.0f);
                found = true;
            }
        }
        float tBot = (baseCenter.y - ray.origin.y) / ray.direction.y;
        if (tBot > tMin && tBot < tClosest) {
            float3 p = ray.origin + ray.direction * tBot;
            float2 v0 = p2.xz - p0.xz;
            float2 v1 = p1.xz - p0.xz;
            float2 v2 = p.xz - p0.xz;
            float dot00 = dot(v0, v0);
            float dot01 = dot(v0, v1);
            float dot02 = dot(v0, v2);
            float dot11 = dot(v1, v1);
            float dot12 = dot(v1, v2);
            float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
            if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
                tClosest = tBot;
                bestNorm = float3(0.0f, -1.0f, 0.0f);
                found = true;
            }
        }
    }

    if (found) {
        tHit = tClosest;
        hitNormal = bestNorm;
        return true;
    }
    return false;
}

inline bool intersectTriangle(
    Ray ray,
    float3 v0, float3 v1, float3 v2,
    float3 n0, float3 n1, float3 n2,
    thread float &tHit, thread float3 &hitNormal
) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(ray.direction, e2);
    float det = dot(e1, pvec);

    if (abs(det) < 1e-8f) return false;
    float invDet = 1.0f / det;

    float3 tvec = ray.origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, e1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    float t = dot(e2, qvec) * invDet;
    if (t < 0.0005f) return false;

    tHit = t;
    float w = 1.0f - u - v;
    float3 N = normalize(w * n0 + u * n1 + v * n2);
    if (dot(N, ray.direction) > 0.0f) N = -N;
    hitNormal = N;
    return true;
}

inline bool intersectTeapotBVHInterval(
    Ray ray,
    device const GPUBVHNode *nodes,
    device const GPUTriangle *triangles,
    uint numNodes,
    float tMin,
    float tMax,
    thread float &tHit,
    thread float3 &hitNormal
) {
    if (numNodes == 0) return false;

    constexpr int kStackSize = 64;
    int stack[kStackSize];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    float tClosest = tMax;
    float3 bestNormal = float3(0.0f);
    bool hitAny = false;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        GPUBVHNode node = nodes[nodeIdx];

        float tBox;
        if (!intersectBoxFast(ray, node.bmin.xyz, node.bmax.xyz, tBox)) continue;
        if (tBox >= tClosest) continue;

        if (node.leftChild < 0) {
            int triCount = -node.leftChild;
            int triStart = node.rightChild;
            for (int i = 0; i < triCount; i++) {
                GPUTriangle tri = triangles[triStart + i];
                float tTri;
                float3 nTri;
                if (intersectTriangle(ray, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz,
                                      tri.n0.xyz, tri.n1.xyz, tri.n2.xyz, tTri, nTri)) {
                    if (tTri >= tMin && tTri < tClosest) {
                        tClosest = tTri;
                        bestNormal = nTri;
                        hitAny = true;
                    }
                }
            }
        } else {
            float tNearL, tNearR;
            bool hitL = intersectBoxFast(ray, nodes[node.leftChild].bmin.xyz, nodes[node.leftChild].bmax.xyz, tNearL);
            bool hitR = intersectBoxFast(ray, nodes[node.rightChild].bmin.xyz, nodes[node.rightChild].bmax.xyz, tNearR);

            if (hitL && hitR && stackPtr + 2 <= kStackSize) {
                // Push the far child first so the near one pops next.
                if (tNearL < tNearR) {
                    stack[stackPtr++] = node.rightChild;
                    stack[stackPtr++] = node.leftChild;
                } else {
                    stack[stackPtr++] = node.leftChild;
                    stack[stackPtr++] = node.rightChild;
                }
            } else if (hitL && stackPtr < kStackSize) {
                stack[stackPtr++] = node.leftChild;
            } else if (hitR && stackPtr < kStackSize) {
                stack[stackPtr++] = node.rightChild;
            }
        }
    }

    if (hitAny) {
        tHit = tClosest;
        hitNormal = bestNormal;
        return true;
    }
    return false;
}

inline bool intersectTeapotBVH(
    Ray ray,
    device const GPUBVHNode *nodes,
    device const GPUTriangle *triangles,
    uint numNodes,
    thread float &tHit,
    thread float3 &hitNormal
) {
    return intersectTeapotBVHInterval(ray, nodes, triangles, numNodes, 0.001f, 1e30f, tHit, hitNormal);
}

inline HitRecord intersectSceneInterval(
    Ray ray,
    bool testGlass,
    device const GPUBVHNode *bvhNodes,
    device const GPUTriangle *triangles,
    uint numNodes,
    float tMin,
    float tMax
) {
    HitRecord hit;
    hit.hit = false;
    hit.distance = tMax;
    hit.isGlass = false;
    hit.roughness = 0.0f;
    hit.objectId = 0;

    float t;
    float3 norm;

    if (abs(ray.direction.z) > 1e-5f) {
        t = (kRoomMaxZ - ray.origin.z) / ray.direction.z;
        if (t >= tMin && t < hit.distance) {
            float3 p = ray.origin + ray.direction * t;
            if (p.x >= kRoomMinX && p.x <= kRoomMaxX && p.y >= kRoomMinY && p.y <= kRoomMaxY) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = float3(0.0f, 0.0f, -1.0f);
                hit.albedo = float3(0.88f, 0.86f, 0.82f);
                hit.roughness = 0.9f;
                hit.isGlass = false;
                hit.objectId = 1;
            }
        }
    }

    if (abs(ray.direction.y) > 1e-5f) {
        t = -ray.origin.y / ray.direction.y;
        if (t >= tMin && t < hit.distance) {
            float3 p = ray.origin + ray.direction * t;
            if (p.x >= kRoomMinX && p.x <= kRoomMaxX && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = float3(0.0f, 1.0f, 0.0f);
                float tileX = fract(p.x * 1.5f);
                float tileZ = fract(p.z * 1.5f);
                float grout = (tileX < 0.03f || tileZ < 0.03f) ? 0.45f : 1.0f;
                hit.albedo = float3(0.72f, 0.70f, 0.65f) * grout;
                hit.roughness = 0.4f;
                hit.isGlass = false;
                hit.objectId = 2;
            }
        }

        t = (kRoomMaxY - ray.origin.y) / ray.direction.y;
        if (t >= tMin && t < hit.distance) {
            float3 p = ray.origin + ray.direction * t;
            if (p.x >= kRoomMinX && p.x <= kRoomMaxX && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = float3(0.0f, -1.0f, 0.0f);
                hit.albedo = float3(0.92f, 0.92f, 0.90f);
                hit.roughness = 0.9f;
                hit.isGlass = false;
                hit.objectId = 3;
            }
        }
    }

    if (abs(ray.direction.x) > 1e-5f) {
        t = (kRoomMinX - ray.origin.x) / ray.direction.x;
        if (t >= tMin && t < hit.distance) {
            float3 p = ray.origin + ray.direction * t;
            if (p.y >= kRoomMinY && p.y <= kRoomMaxY && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                if (p.y >= kWinMinY && p.y <= kWinMaxY && p.z >= kWinMinZ && p.z <= kWinMaxZ) {
                    bool isMuntin = (abs(p.y - 1.80f) < 0.045f) ||
                                    (abs(p.z - 0.00f) < 0.045f) ||
                                    (abs(p.z - 1.05f) < 0.040f) ||
                                    (abs(p.z + 1.05f) < 0.040f) ||
                                    (abs(p.y - kWinMinY) < 0.06f) ||
                                    (abs(p.y - kWinMaxY) < 0.06f) ||
                                    (abs(p.z - kWinMinZ) < 0.06f) ||
                                    (abs(p.z - kWinMaxZ) < 0.06f);
                    if (isMuntin) {
                        hit.hit = true;
                        hit.distance = t;
                        hit.position = p;
                        hit.normal = float3(1.0f, 0.0f, 0.0f);
                        hit.albedo = float3(0.24f, 0.16f, 0.10f);
                        hit.roughness = 0.6f;
                        hit.isGlass = false;
                        hit.objectId = 6;
                    }
                } else {
                    hit.hit = true;
                    hit.distance = t;
                    hit.position = p;
                    hit.normal = float3(1.0f, 0.0f, 0.0f);
                    hit.albedo = float3(0.85f, 0.22f, 0.20f);
                    hit.roughness = 0.85f;
                    hit.isGlass = false;
                    hit.objectId = 4;
                }
            }
        }

        t = (kRoomMaxX - ray.origin.x) / ray.direction.x;
        if (t >= tMin && t < hit.distance) {
            float3 p = ray.origin + ray.direction * t;
            if (p.y >= kRoomMinY && p.y <= kRoomMaxY && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = float3(-1.0f, 0.0f, 0.0f);
                hit.albedo = float3(0.18f, 0.75f, 0.40f);
                hit.roughness = 0.85f;
                hit.isGlass = false;
                hit.objectId = 5;
            }
        }
    }

    if (testGlass) {
        if (intersectTeapotBVHInterval(ray, bvhNodes, triangles, numNodes, tMin, hit.distance, t, norm)) {
            hit.hit = true;
            hit.distance = t;
            hit.position = ray.origin + ray.direction * t;
            hit.normal = norm;
            hit.albedo = float3(1.0f);
            hit.roughness = 0.0f;
            hit.isGlass = true;
            hit.objectId = 10;
            hit.ior = 1.52f;
            hit.dispersion = 0.025f;
            hit.absorption = float3(0.04f, 0.04f, 0.04f);
        }

        if (intersectSphere(ray, kSphereCenter, kSphereRadius, tMin, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = float3(1.0f);
                hit.roughness = 0.0f;
                hit.isGlass = true;
                hit.objectId = 11;
                hit.ior = 1.62f;
                hit.dispersion = 0.040f;
                hit.absorption = float3(0.02f, 0.02f, 0.02f);
            }
        }

        if (intersectCylinder(ray, kCylinderCenter, kCylinderRadius, kCylinderHeight, tMin, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = float3(1.0f);
                hit.roughness = 0.0f;
                hit.isGlass = true;
                hit.objectId = 12;
                hit.ior = 1.50f;
                hit.dispersion = 0.010f;
                hit.absorption = float3(1.20f, 0.15f, 0.90f) * 2.2f;
            }
        }

        if (intersectTriangularPrism(ray, kPrismCenter, kPrismSide, kPrismHeight, tMin, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = float3(1.0f);
                hit.roughness = 0.0f;
                hit.isGlass = true;
                hit.objectId = 13;
                hit.ior = 1.58f;
                hit.dispersion = 0.055f;
                hit.absorption = float3(0.03f, 0.03f, 0.03f);
            }
        }

        float3 slabMin = float3( 1.00f, 0.0f, 0.35f);
        float3 slabMax = float3( 1.70f, 0.06f, 1.05f);
        if (intersectBox(ray, slabMin, slabMax, tMin, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = float3(0.92f, 0.94f, 0.96f);
                hit.roughness = 0.35f;
                hit.isGlass = true;
                hit.objectId = 14;
                hit.ior = 1.52f;
                hit.dispersion = 0.015f;
                hit.absorption = float3(0.2f, 0.15f, 0.1f);
            }
        }
    }

    return hit;
}

inline HitRecord intersectScene(
    Ray ray,
    bool testGlass,
    device const GPUBVHNode *bvhNodes,
    device const GPUTriangle *triangles,
    uint numNodes
) {
    return intersectSceneInterval(ray, testGlass, bvhNodes, triangles, numNodes, 0.001f, 1e30f);
}

inline float3 getSkyRadiance(float3 direction, float3 sunDir) {
    float sunDot = max(0.0f, dot(direction, sunDir));
    float3 zenithColor = float3(0.40f, 0.62f, 0.95f);
    float3 horizonColor = float3(0.78f, 0.85f, 0.95f);
    float3 sunGlowColor = float3(1.0f, 0.92f, 0.75f);

    float hFactor = saturate(direction.y * 1.5f);
    float3 sky = mix(horizonColor, zenithColor, hFactor);
    float sunDisc = pow(sunDot, 128.0f) * 4.0f + pow(sunDot, 1024.0f) * 20.0f;
    return sky + sunGlowColor * sunDisc;
}

struct Basis {
    float3 tangent;
    float3 bitangent;
    float3 normal;

    float3 toWorld(float3 v) const {
        return v.x * tangent + v.y * bitangent + v.z * normal;
    }
};

inline Basis makeTBN(float3 N) {
    Basis b;
    b.normal = N;
    if (abs(N.y) > 0.999f) {
        b.tangent = float3(1.0f, 0.0f, 0.0f);
        b.bitangent = float3(0.0f, 0.0f, 1.0f);
    } else {
        b.tangent = normalize(cross(N, float3(0.0f, 1.0f, 0.0f)));
        b.bitangent = cross(b.tangent, N);
    }
    return b;
}

constant uint kAtlasSurfaceWidth = 64;
constant uint kAtlasSurfaceHeight = 64;
constant uint kNumSurfaces = 5;

inline void getSurfaceGeometry(uint surfaceId, float2 uv, thread float3 &pos, thread float3 &nor) {
    float u = clamp(uv.x, 0.001f, 0.999f);
    float v = clamp(uv.y, 0.001f, 0.999f);
    if (surfaceId == 0) {
        pos = float3(mix(kRoomMinX, kRoomMaxX, u), 0.002f, mix(kRoomMinZ, kRoomMaxZ, v));
        nor = float3(0.0f, 1.0f, 0.0f);
    } else if (surfaceId == 1) {
        pos = float3(mix(kRoomMinX, kRoomMaxX, u), kRoomMaxY - 0.002f, mix(kRoomMinZ, kRoomMaxZ, v));
        nor = float3(0.0f, -1.0f, 0.0f);
    } else if (surfaceId == 2) {
        pos = float3(mix(kRoomMinX, kRoomMaxX, u), mix(kRoomMinY + 0.002f, kRoomMaxY - 0.002f, v), kRoomMaxZ - 0.002f);
        nor = float3(0.0f, 0.0f, -1.0f);
    } else if (surfaceId == 3) {
        pos = float3(kRoomMinX + 0.002f, mix(kRoomMinY + 0.002f, kRoomMaxY - 0.002f, v), mix(kRoomMinZ, kRoomMaxZ, u));
        nor = float3(1.0f, 0.0f, 0.0f);
    } else {
        pos = float3(kRoomMaxX - 0.002f, mix(kRoomMinY + 0.002f, kRoomMaxY - 0.002f, v), mix(kRoomMinZ, kRoomMaxZ, u));
        nor = float3(-1.0f, 0.0f, 0.0f);
    }
}

inline float3 sampleSurfaceAtlas(
    uint surfaceId,
    float u,
    float v,
    texture2d<float, access::sample> atlas
) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float uTexel = (clamp(u, 0.001f, 0.999f) * float(kAtlasSurfaceWidth - 1) + 0.5f) / float(kAtlasSurfaceWidth * kNumSurfaces);
    float atlasU = (float(surfaceId) / float(kNumSurfaces)) + uTexel;
    float atlasV = (clamp(v, 0.001f, 0.999f) * float(kAtlasSurfaceHeight - 1) + 0.5f) / float(kAtlasSurfaceHeight);
    return atlas.sample(s, float2(atlasU, atlasV)).rgb;
}

inline float3 sampleIrradianceAtlas(
    HitRecord hit,
    texture2d<float, access::sample> irradianceAtlas
) {
    if (hit.objectId == 2) {
        float u = (hit.position.x - kRoomMinX) / (kRoomMaxX - kRoomMinX);
        float v = (hit.position.z - kRoomMinZ) / (kRoomMaxZ - kRoomMinZ);
        return sampleSurfaceAtlas(0, u, v, irradianceAtlas);
    } else if (hit.objectId == 3) {
        float u = (hit.position.x - kRoomMinX) / (kRoomMaxX - kRoomMinX);
        float v = (hit.position.z - kRoomMinZ) / (kRoomMaxZ - kRoomMinZ);
        return sampleSurfaceAtlas(1, u, v, irradianceAtlas);
    } else if (hit.objectId == 1) {
        float u = (hit.position.x - kRoomMinX) / (kRoomMaxX - kRoomMinX);
        float v = (hit.position.y - kRoomMinY) / (kRoomMaxY - kRoomMinY);
        return sampleSurfaceAtlas(2, u, v, irradianceAtlas);
    } else if (hit.objectId == 4 || hit.objectId == 6) {
        float u = (hit.position.z - kRoomMinZ) / (kRoomMaxZ - kRoomMinZ);
        float v = (hit.position.y - kRoomMinY) / (kRoomMaxY - kRoomMinY);
        return sampleSurfaceAtlas(3, u, v, irradianceAtlas);
    } else if (hit.objectId == 5) {
        float u = (hit.position.z - kRoomMinZ) / (kRoomMaxZ - kRoomMinZ);
        float v = (hit.position.y - kRoomMinY) / (kRoomMaxY - kRoomMinY);
        return sampleSurfaceAtlas(4, u, v, irradianceAtlas);
    }

    // Glass objects have no atlas slot of their own, so blend the five wall
    // probes by how much of each the shading normal faces.
    float u_xz = (hit.position.x - kRoomMinX) / (kRoomMaxX - kRoomMinX);
    float v_xz = (hit.position.z - kRoomMinZ) / (kRoomMaxZ - kRoomMinZ);

    float u_xy = (hit.position.x - kRoomMinX) / (kRoomMaxX - kRoomMinX);
    float v_xy = (hit.position.y - kRoomMinY) / (kRoomMaxY - kRoomMinY);

    float u_yz = (hit.position.z - kRoomMinZ) / (kRoomMaxZ - kRoomMinZ);
    float v_yz = (hit.position.y - kRoomMinY) / (kRoomMaxY - kRoomMinY);

    float3 irrFloor = sampleSurfaceAtlas(0, u_xz, v_xz, irradianceAtlas);
    float3 irrCeil  = sampleSurfaceAtlas(1, u_xz, v_xz, irradianceAtlas);
    float3 irrBack  = sampleSurfaceAtlas(2, u_xy, v_xy, irradianceAtlas);
    float3 irrLeft  = sampleSurfaceAtlas(3, u_yz, v_yz, irradianceAtlas);
    float3 irrRight = sampleSurfaceAtlas(4, u_yz, v_yz, irradianceAtlas);

    float wFloor = max(0.0f, -hit.normal.y);
    float wCeil  = max(0.0f,  hit.normal.y);
    float wBack  = max(0.0f,  hit.normal.z);
    float wLeft  = max(0.0f, -hit.normal.x);
    float wRight = max(0.0f,  hit.normal.x);

    float wSum = wFloor + wCeil + wBack + wLeft + wRight + 1e-4f;
    return (wFloor * irrFloor + wCeil * irrCeil + wBack * irrBack + wLeft * irrLeft + wRight * irrRight) / wSum;
}

struct CascadeLevelParams {
    float tMin;
    float tMax;
    uint raysThisLevel;
    uint raysUpperLevel;          // 0 marks the terminal (farthest) cascade: nothing to merge from.
    uint probesPerAxisThisLevel;
    uint probesPerAxisUpperLevel; // unused when raysUpperLevel == 0
    float skyBoost;
};

// Cosine-weighted Fibonacci direction `index` out of `count`, in the probe's
// tangent frame. Level N+1 has 4x the directions of level N (see
// cascadeGatherKernel), so index i here corresponds to [4i, 4i+3] one level up.
inline float3 cascadeDirection(Basis tbn, int index, int count, float jitter) {
    float cosTheta = sqrt(max(0.0f, 1.0f - (float(index) + 0.5f) / float(count)));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = float(index) * 2.399963229728f + jitter;
    return tbn.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));
}

// A cascade's probes sit at the centre of a probesPerAxis x probesPerAxis grid
// over the surface, one probe per thread row rather than one per atlas texel.
inline void cascadeProbeAt(uint surfaceId, uint probeX, uint probeY, uint probesPerAxis,
                           thread float3 &origin, thread Basis &tbn, thread float2 &uv) {
    uv = (float2(probeX, probeY) + 0.5f) / float(probesPerAxis);
    float3 pos, nor;
    getSurfaceGeometry(surfaceId, uv, pos, nor);
    tbn = makeTBN(nor);
    origin = pos + nor * 0.004f;
}

// Manual bilinear fetch across a coarser cascade's probe grid, at a fixed
// direction index. Hardware texture filtering can't be used here because
// probes and directions are packed along the same texture axis, and blending
// across a direction boundary would mix unrelated rays.
inline float3 sampleCascadeBilinear(
    texture2d_array<float, access::read> cascade,
    uint surfaceId,
    float2 uv,
    uint probesPerAxis,
    uint rays,
    uint dirIndex
) {
    float gx = uv.x * float(probesPerAxis) - 0.5f;
    float gy = uv.y * float(probesPerAxis) - 0.5f;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    float fx = gx - float(x0);
    float fy = gy - float(y0);

    int maxIdx = int(probesPerAxis) - 1;
    int x1 = clamp(x0 + 1, 0, maxIdx);
    int y1 = clamp(y0 + 1, 0, maxIdx);
    x0 = clamp(x0, 0, maxIdx);
    y0 = clamp(y0, 0, maxIdx);

    float3 c00 = cascade.read(uint2(uint(x0) * rays + dirIndex, uint(y0)), surfaceId).rgb;
    float3 c10 = cascade.read(uint2(uint(x1) * rays + dirIndex, uint(y0)), surfaceId).rgb;
    float3 c01 = cascade.read(uint2(uint(x0) * rays + dirIndex, uint(y1)), surfaceId).rgb;
    float3 c11 = cascade.read(uint2(uint(x1) * rays + dirIndex, uint(y1)), surfaceId).rgb;

    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

// Traces ray strictly within distance interval [tMin, tMax] with temporal multi-bounce
inline float4 traceCascadeInterval(
    float3 origin,
    float3 dir,
    float tMin,
    float tMax,
    float3 sunDir,
    float3 sunCol,
    float sunInt,
    uint numNodes,
    device const GPUBVHNode *bvhNodes,
    device const GPUTriangle *triangles,
    texture2d<float, access::sample> prevAtlas
) {
    Ray probeRay;
    probeRay.origin = origin;
    probeRay.direction = dir;
    HitRecord hit = intersectSceneInterval(probeRay, true, bvhNodes, triangles, numNodes, tMin, tMax);
    if (hit.hit) {
        float NdotL = max(0.0f, dot(hit.normal, sunDir));
        float3 directSun = float3(0.0f);
        if (NdotL > 0.0f) {
            Ray sRay;
            sRay.origin = hit.position + hit.normal * 0.002f;
            sRay.direction = sunDir;
            HitRecord sHit = intersectSceneInterval(sRay, false, bvhNodes, triangles, numNodes, 0.001f, 100.0f);
            if (!sHit.hit) {
                directSun = sunCol * (sunInt * NdotL);
            }
        }
        float3 bouncedGI = sampleIrradianceAtlas(hit, prevAtlas);
        float3 hitRad = (directSun + bouncedGI) * hit.albedo;

        float tNorm = clamp((hit.distance - tMin) / max(1e-4f, tMax - tMin), 0.0f, 1.0f);
        float boundaryFade = smoothstep(0.85f, 1.0f, tNorm);
        return float4(hitRad, boundaryFade);
    }
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

// One thread per (probe, direction) at this cascade level, dispatched once
// per level, far-to-near (3, 2, 1, 0). Each level traces its own probe grid
// exactly once and reads the level above through sampleCascadeBilinear rather
// than retracing it, so a coarse cascade costs what its own probe count and
// ray count say it costs, not what the finest level below it costs.
kernel void cascadeGatherKernel(
    uint3 tid [[thread_position_in_grid]],
    texture2d_array<float, access::write> outCascade [[texture(0)]],
    texture2d_array<float, access::read> upperCascade [[texture(1)]],
    texture2d<float, access::sample> prevAtlas [[texture(2)]],
    constant GlassUniforms &uniforms [[buffer(0)]],
    constant CascadeLevelParams &level [[buffer(1)]],
    device const GPUBVHNode *bvhNodes [[buffer(2)]],
    device const GPUTriangle *triangles [[buffer(3)]]
) {
    uint perSurfaceWidth = level.probesPerAxisThisLevel * level.raysThisLevel;
    if (tid.x >= perSurfaceWidth || tid.y >= level.probesPerAxisThisLevel || tid.z >= kNumSurfaces) return;

    uint surfaceId = tid.z;
    uint probeX = tid.x / level.raysThisLevel;
    uint dirIndex = tid.x % level.raysThisLevel;
    uint probeY = tid.y;

    float3 origin;
    Basis tbn;
    float2 uv;
    cascadeProbeAt(surfaceId, probeX, probeY, level.probesPerAxisThisLevel, origin, tbn, uv);

    // Rotates a little every frame so the temporal blend in
    // cascadeIntegrateKernel averages away noise instead of freezing it.
    float jitterSeed = float(surfaceId) * 37.0f + float(uniforms.frameIndex) * 0.6180339887f;
    float jitter = fract(sin(dot(float2(probeX, probeY) + jitterSeed, float2(12.9898f, 78.233f))) * 43758.5453f) * (2.0f * kPi);

    float3 sunDir = normalize(uniforms.sunDirection);
    float3 dir = cascadeDirection(tbn, int(dirIndex), int(level.raysThisLevel), jitter);
    float4 seg = traceCascadeInterval(origin, dir, level.tMin, level.tMax,
                                      sunDir, uniforms.sunColor, uniforms.sunIntensity,
                                      uniforms.numTeapotNodes, bvhNodes, triangles, prevAtlas);

    float3 result;
    if (level.raysUpperLevel == 0u) {
        result = seg.w > 0.0f ? seg.xyz + seg.w * getSkyRadiance(dir, sunDir) * level.skyBoost : seg.xyz;
    } else if (seg.w <= 0.0f) {
        result = seg.xyz;
    } else {
        float3 fromUpper = float3(0.0f);
        uint upperBase = dirIndex * 4u;
        for (uint k = 0u; k < 4u; k++) {
            fromUpper += 0.25f * sampleCascadeBilinear(upperCascade, surfaceId, uv,
                                                        level.probesPerAxisUpperLevel, level.raysUpperLevel,
                                                        upperBase + k);
        }
        result = seg.xyz + seg.w * fromUpper;
    }

    outCascade.write(float4(result, 1.0f), uint2(tid.x, tid.y), surfaceId);
}

// Cascade 0 has exactly kAtlasSurfaceWidth probes per axis, one per output
// atlas texel, so folding it into irradiance is a plain average over its
// directions plus the existing history blend.
kernel void cascadeIntegrateKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::read_write> irradianceAtlas [[texture(0)]],
    texture2d_array<float, access::read> cascade0 [[texture(1)]],
    constant GlassUniforms &uniforms [[buffer(0)]],
    constant CascadeLevelParams &level [[buffer(1)]]
) {
    if (tid.x >= kAtlasSurfaceWidth * kNumSurfaces || tid.y >= kAtlasSurfaceHeight) return;

    uint surfaceId = tid.x / kAtlasSurfaceWidth;
    uint probeX = tid.x % kAtlasSurfaceWidth;
    uint probeY = tid.y;

    uint base = probeX * level.raysThisLevel;

    float3 sum = float3(0.0f);
    for (uint i = 0u; i < level.raysThisLevel; i++) {
        sum += cascade0.read(uint2(base + i, probeY), surfaceId).rgb;
    }
    float3 newIrradiance = sum / float(level.raysThisLevel);

    // Blend against the unfiltered history. Feeding the blurred atlas back in
    // would re-apply the spatial filter every frame and creep towards mush.
    if (uniforms.frameIndex > 1 && (uniforms.ablationMask & kAblTemporal) != 0u) {
        newIrradiance = mix(newIrradiance, irradianceAtlas.read(tid).rgb, 0.70f);
    }
    irradianceAtlas.write(float4(newIrradiance, 1.0f), tid);
}

kernel void filterIrradianceAtlasKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::read> inAtlas [[texture(0)]],
    texture2d<float, access::write> outAtlas [[texture(1)]]
) {
    if (tid.x >= kAtlasSurfaceWidth * kNumSurfaces || tid.y >= kAtlasSurfaceHeight) return;

    uint surfaceId = tid.x / kAtlasSurfaceWidth;
    uint lx = tid.x % kAtlasSurfaceWidth;
    uint ly = tid.y;

    const int kRadius = 3;
    const float sigma = 2.0f;
    const float twoSigma2 = 2.0f * sigma * sigma;

    float3 accum = float3(0.0f);
    float weightSum = 0.0f;

    for (int dy = -kRadius; dy <= kRadius; dy++) {
        int sy = clamp(int(ly) + dy, 0, int(kAtlasSurfaceHeight) - 1);
        for (int dx = -kRadius; dx <= kRadius; dx++) {
            int sx = clamp(int(lx) + dx, 0, int(kAtlasSurfaceWidth) - 1);
            float dist2 = float(dx * dx + dy * dy);
            float w = exp(-dist2 / twoSigma2);

            uint2 sampleCoord = uint2(surfaceId * kAtlasSurfaceWidth + uint(sx), uint(sy));
            accum += inAtlas.read(sampleCoord).rgb * w;
            weightSum += w;
        }
    }

    outAtlas.write(float4(accum / max(1e-5f, weightSum), 1.0f), tid);
}

inline float3 evaluateSurfaceRadiance(
    HitRecord hit,
    Ray ray,
    constant GlassUniforms &uniforms,
    bool useCascadeGI,
    texture2d<float, access::sample> causticTexture,
    texture2d<float, access::sample> irradianceAtlas,
    device const GPUBVHNode *bvhNodes,
    device const GPUTriangle *triangles
) {
    if (!hit.hit) {
        return getSkyRadiance(ray.direction, normalize(uniforms.sunDirection));
    }

    float3 L = normalize(uniforms.sunDirection);
    float NdotL = max(0.0f, dot(hit.normal, L));
    float3 directSun = float3(0.0f);

    if (NdotL > 0.0f) {
        Ray shadowRay;
        shadowRay.origin = hit.position + hit.normal * 0.002f;
        shadowRay.direction = L;

        HitRecord shadowHit = intersectScene(shadowRay, false, bvhNodes, triangles, 0);
        if (!shadowHit.hit) {
            directSun = uniforms.sunColor * (uniforms.sunIntensity * NdotL);
        }
    }

    float3 causticRad = float3(0.0f);
    if (hit.objectId == 2 && (uniforms.ablationMask & kAblCaustics) != 0u) {
        float uFloor = (hit.position.x - kFloorMinX) / (kFloorMaxX - kFloorMinX);
        float vFloor = (hit.position.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ);
        if (uFloor >= 0.0f && uFloor <= 1.0f && vFloor >= 0.0f && vFloor <= 1.0f) {
            constexpr sampler causticSampler(coord::normalized, filter::linear, address::clamp_to_edge);
            causticRad = causticTexture.sample(causticSampler, float2(uFloor, vFloor)).rgb;
        }
    }

    float3 indirectGI = float3(0.0f);
    if (useCascadeGI && (uniforms.ablationMask & kAblCascadeGI) != 0u) {
        float3 E = sampleIrradianceAtlas(hit, irradianceAtlas);
        indirectGI = E * hit.albedo;
    } else {
        float3 baselineAmbient = float3(0.04f);
        indirectGI = baselineAmbient * hit.albedo;
    }

    return (directSun + causticRad) * hit.albedo + indirectGI;
}

inline float3 toneMapACES(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Photon flux is accumulated in fixed point so the splat can be atomic.
constant float kFluxFixedScale = 1000000000.0f;

// Bilinear splat of a single channel's flux into the floor accumulation buffer.
inline void splatFlux(device atomic_uint *buffer, float2 floorUV, uint channel, float flux) {
    float gx = floorUV.x * float(kCausticRes) - 0.5f;
    float gy = floorUV.y * float(kCausticRes) - 0.5f;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    float fx = gx - float(x0);
    float fy = gy - float(y0);

    float weights[4] = { (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy),
                         (1.0f - fx) * fy,          fx * fy };

    for (int k = 0; k < 4; k++) {
        int x = x0 + (k & 1);
        int y = y0 + (k >> 1);
        if (x < 0 || y < 0 || x >= int(kCausticRes) || y >= int(kCausticRes)) continue;
        uint idx = (uint(y) * kCausticRes + uint(x)) * 4u + channel;
        atomic_fetch_add_explicit(&buffer[idx], uint(flux * weights[k] * kFluxFixedScale), memory_order_relaxed);
    }
}

// Continues an exit ray down to the floor plane; false if it misses the slab.
inline bool hitFloorUV(float3 origin, float3 dir, thread float2 &floorUV) {
    if (dir.y >= -1e-4f) return false;
    float t = -origin.y / dir.y;
    if (t <= 0.0f) return false;

    float3 p = origin + dir * t;
    if (p.x < kFloorMinX || p.x > kFloorMaxX || p.z < kFloorMinZ || p.z > kFloorMaxZ) return false;

    floorUV = float2((p.x - kFloorMinX) / (kFloorMaxX - kFloorMinX),
                     (p.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ));
    return true;
}

// Snell exit from inside a dielectric. N must already point into the medium.
inline bool exitRefract(float3 incident, float3 N, float ior, float cosInside, thread float3 &outDir) {
    float sin2Out = (1.0f - cosInside * cosInside) * (ior * ior);
    if (sin2Out >= 1.0f) return false;
    outDir = ior * incident - (ior * cosInside - sqrt(1.0f - sin2Out)) * N;
    return true;
}

// One photon per thread. The 2048x2048 grid is split into four 1024x1024
// quadrants, one per glass object, so all four share a single dispatch.
kernel void generateCausticsKernel(
    uint2 tid [[thread_position_in_grid]],
    device atomic_uint *causticBuffer [[buffer(0)]],
    constant GlassUniforms &uniforms [[buffer(1)]],
    device const GPUBVHNode *bvhNodes [[buffer(2)]],
    device const GPUTriangle *triangles [[buffer(3)]]
) {
    if (tid.x >= 2048 || tid.y >= 2048) return;

    uint quadrant = (tid.x / 1024) + (tid.y / 1024) * 2;
    float2 uv = (float2(tid.x % 1024, tid.y % 1024) + 0.5f) / 1024.0f;

    float3 L = normalize(uniforms.sunDirection);
    float3 lightDir = -L;

    // Emitter disc basis, perpendicular to the sun.
    float3 up = abs(L.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 uAxis = normalize(cross(L, up));
    float3 vAxis = cross(L, uAxis);

    if (quadrant == 0) {
        // Sphere: entry and exit are both analytic, so no tracing is needed.
        float sx = (uv.x - 0.5f) * (2.0f * kSphereRadius);
        float sy = (uv.y - 0.5f) * (2.0f * kSphereRadius);
        float s2 = sx * sx + sy * sy;
        if (s2 >= kSphereRadius * kSphereRadius) return;

        float sz = sqrt(max(0.0f, kSphereRadius * kSphereRadius - s2));
        float3 P1 = kSphereCenter + sx * uAxis + sy * vAxis + sz * L;
        float3 N1 = (P1 - kSphereCenter) / kSphereRadius;

        float cosEntry = clamp(-dot(lightDir, N1), 0.0f, 1.0f);
        float rayWeight = (4.0f * kSphereRadius * kSphereRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        // Matches the sphere's scene material (ior 1.62, dispersion 0.040).
        float disp = (uniforms.renderMode == 3) ? 0.060f : 0.040f;
        float3 iors = 1.62f + float3(-disp, 0.0f, disp);

        for (uint ch = 0; ch < 3; ch++) {
            float eta = iors[ch];

            float sin2Inside = (1.0f - cosEntry * cosEntry) / (eta * eta);
            if (sin2Inside >= 1.0f) continue;
            float cosInside = sqrt(1.0f - sin2Inside);

            float3 D1 = (lightDir / eta) + (cosEntry / eta - cosInside) * N1;
            float internalDist = 2.0f * kSphereRadius * cosInside;
            float3 P2 = P1 + D1 * internalDist;
            float3 N2 = (P2 - kSphereCenter) / kSphereRadius;

            float cosExit = clamp(dot(D1, N2), 0.0f, 1.0f);
            float3 D2;
            if (!exitRefract(D1, N2, eta, cosExit, D2)) continue;

            float2 floorUV;
            if (!hitFloorUV(P2, D2, floorUV)) continue;

            float flux = rayWeight * uniforms.sunColor[ch]
                       * (1.0f - dielectricFresnel(cosEntry, 1.0f, eta))
                       * (1.0f - dielectricFresnel(cosExit, eta, 1.0f))
                       * exp(-0.02f * internalDist);
            splatFlux(causticBuffer, floorUV, ch, flux);
        }
        return;
    }

    if (quadrant == 1) {
        // Teapot: two BVH queries, one for the entry hull and one for the exit.
        if (uniforms.numTeapotNodes == 0) return;

        // Covers the mesh's XZ diagonal (bounds are +-0.664 x +-0.413).
        const float objRadius = 0.78f;
        float sx = (uv.x - 0.5f) * (2.0f * objRadius);
        float sy = (uv.y - 0.5f) * (2.0f * objRadius);
        if (sx * sx + sy * sy > objRadius * objRadius) return;

        Ray photonRay;
        photonRay.origin = float3(0.0f, 0.325f, 0.0f) + sx * uAxis + sy * vAxis - lightDir * 3.5f;
        photonRay.direction = lightDir;

        float tEntry;
        float3 nEntry;
        if (!intersectTeapotBVH(photonRay, bvhNodes, triangles, uniforms.numTeapotNodes, tEntry, nEntry)) return;

        float3 P1 = photonRay.origin + photonRay.direction * tEntry;
        float cosEntry = clamp(-dot(lightDir, nEntry), 0.0f, 1.0f);
        float rayWeight = (4.0f * objRadius * objRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        const float objIor = 1.52f;
        float3 D1;
        if (!refractRay(lightDir, nEntry, 1.0f / objIor, D1)) return;

        Ray insideRay;
        insideRay.origin = P1 + D1 * 0.005f;
        insideRay.direction = D1;

        float tExit;
        float3 nExit;
        if (!intersectTeapotBVH(insideRay, bvhNodes, triangles, uniforms.numTeapotNodes, tExit, nExit)) return;
        if (tExit < 0.002f) return;

        float3 P2 = insideRay.origin + insideRay.direction * tExit;
        float3 N2 = -nExit;
        float cosExit = clamp(dot(D1, N2), 0.0f, 1.0f);

        float3 baseFlux = rayWeight * uniforms.sunColor
                        * (1.0f - dielectricFresnel(cosEntry, 1.0f, objIor))
                        * (1.0f - dielectricFresnel(cosExit, objIor, 1.0f))
                        * beerLambertAbsorption(float3(0.04f), tExit);

        float disp = (uniforms.renderMode == 3) ? 0.0375f : 0.0f;
        float3 iors = objIor + float3(-disp, 0.0f, disp);

        for (uint ch = 0; ch < 3; ch++) {
            float3 D2;
            if (!exitRefract(D1, N2, iors[ch], cosExit, D2)) continue;

            float2 floorUV;
            if (!hitFloorUV(P2, D2, floorUV)) continue;
            splatFlux(causticBuffer, floorUV, ch, baseFlux[ch]);
        }
        return;
    }

    if (quadrant == 2) {
        // Cylinder: strongly absorbing, so the caustic is tinted magenta.
        const float objRadius = kCylinderRadius * 1.15f;
        float sx = (uv.x - 0.5f) * (2.0f * objRadius);
        float sy = (uv.y - 0.5f) * (2.0f * objRadius);
        if (sx * sx + sy * sy > objRadius * objRadius) return;

        float3 cylCenter = kCylinderCenter + float3(0.0f, kCylinderHeight * 0.5f, 0.0f);
        Ray photonRay;
        photonRay.origin = cylCenter + sx * uAxis + sy * vAxis - lightDir * 3.5f;
        photonRay.direction = lightDir;

        float tEntry;
        float3 nEntry;
        if (!intersectCylinder(photonRay, kCylinderCenter, kCylinderRadius, kCylinderHeight, 0.001f, tEntry, nEntry)) return;

        float3 P1 = photonRay.origin + photonRay.direction * tEntry;
        float cosEntry = clamp(-dot(lightDir, nEntry), 0.0f, 1.0f);
        float rayWeight = (4.0f * objRadius * objRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        const float objIor = 1.50f;
        float3 D1;
        if (!refractRay(lightDir, nEntry, 1.0f / objIor, D1)) return;

        Ray insideRay;
        insideRay.origin = P1 + D1 * 0.005f;
        insideRay.direction = D1;

        float tExit;
        float3 nExit;
        if (!intersectCylinder(insideRay, kCylinderCenter, kCylinderRadius, kCylinderHeight, 0.001f, tExit, nExit)) return;
        if (tExit < 0.002f) return;

        float3 P2 = insideRay.origin + insideRay.direction * tExit;
        float3 N2 = -nExit;
        float cosExit = clamp(dot(D1, N2), 0.0f, 1.0f);

        float3 D2;
        if (!exitRefract(D1, N2, objIor, cosExit, D2)) return;

        float2 floorUV;
        if (!hitFloorUV(P2, D2, floorUV)) return;

        float3 flux = rayWeight * uniforms.sunColor
                    * (1.0f - dielectricFresnel(cosEntry, 1.0f, objIor))
                    * (1.0f - dielectricFresnel(cosExit, objIor, 1.0f))
                    * beerLambertAbsorption(float3(1.2f, 0.15f, 0.9f) * 2.2f, tExit);

        for (uint ch = 0; ch < 3; ch++) splatFlux(causticBuffer, floorUV, ch, flux[ch]);
        return;
    }

    {
        // Prism: each wavelength refracts at its own angle on entry as well as
        // exit, so the three channels have to be traced separately.
        const float objRadius = kPrismSide * 0.75f;
        float sx = (uv.x - 0.5f) * (2.0f * objRadius);
        float sy = (uv.y - 0.5f) * (2.0f * objRadius);
        if (sx * sx + sy * sy > objRadius * objRadius) return;

        float3 prismCenter = kPrismCenter + float3(0.0f, kPrismHeight * 0.5f, 0.0f);
        Ray photonRay;
        photonRay.origin = prismCenter + sx * uAxis + sy * vAxis - lightDir * 3.5f;
        photonRay.direction = lightDir;

        float tEntry;
        float3 nEntry;
        if (!intersectTriangularPrism(photonRay, kPrismCenter, kPrismSide, kPrismHeight, 0.001f, tEntry, nEntry)) return;

        float3 P1 = photonRay.origin + photonRay.direction * tEntry;
        float cosEntry = clamp(-dot(lightDir, nEntry), 0.0f, 1.0f);
        float rayWeight = (4.0f * objRadius * objRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        float3 iors = 1.58f + float3(-0.055f, 0.0f, 0.055f);

        for (uint ch = 0; ch < 3; ch++) {
            float eta = iors[ch];

            float3 D1;
            if (!refractRay(lightDir, nEntry, 1.0f / eta, D1)) continue;

            Ray insideRay;
            insideRay.origin = P1 + D1 * 0.005f;
            insideRay.direction = D1;

            float tExit;
            float3 nExit;
            if (!intersectTriangularPrism(insideRay, kPrismCenter, kPrismSide, kPrismHeight, 0.001f, tExit, nExit)) continue;
            if (tExit < 0.002f) continue;

            float3 P2 = insideRay.origin + insideRay.direction * tExit;
            float3 N2 = -nExit;
            float cosExit = clamp(dot(D1, N2), 0.0f, 1.0f);

            float3 D2;
            if (!exitRefract(D1, N2, eta, cosExit, D2)) continue;

            float2 floorUV;
            if (!hitFloorUV(P2, D2, floorUV)) continue;

            // 1.5x compensates for splitting one photon across three narrow bands.
            float flux = rayWeight * uniforms.sunColor[ch] * 1.5f
                       * (1.0f - dielectricFresnel(cosEntry, 1.0f, eta))
                       * (1.0f - dielectricFresnel(cosExit, eta, 1.0f));
            splatFlux(causticBuffer, floorUV, ch, flux);
        }
    }
}

kernel void filterCausticsKernel(
    uint2 tid [[thread_position_in_grid]],
    device const atomic_uint *causticBuffer [[buffer(0)]],
    texture2d<float, access::write> causticTexture [[texture(0)]],
    constant GlassUniforms &uniforms [[buffer(1)]]
) {
    if (tid.x >= kCausticRes || tid.y >= kCausticRes) return;

    float pixelArea = ((kFloorMaxX - kFloorMinX) / float(kCausticRes)) *
                      ((kFloorMaxZ - kFloorMinZ) / float(kCausticRes));
    float invFixedPoint = 1.0f / (kFluxFixedScale * pixelArea);

    if (uniforms.renderMode != 2) {
        uint baseIdx = (tid.y * kCausticRes + tid.x) * 4u;
        float r = float(atomic_load_explicit(&causticBuffer[baseIdx + 0u], memory_order_relaxed)) * invFixedPoint;
        float g = float(atomic_load_explicit(&causticBuffer[baseIdx + 1u], memory_order_relaxed)) * invFixedPoint;
        float b = float(atomic_load_explicit(&causticBuffer[baseIdx + 2u], memory_order_relaxed)) * invFixedPoint;
        causticTexture.write(float4(r, g, b, 1.0f), tid);
    } else {
        int filterRadius = int(uniforms.glassRoughness * 18.0f + 2.0f);
        float filterWeightSum = 0.0f;
        float3 accumIrradiance = float3(0.0f);

        float sigma = float(filterRadius) * 0.5f;
        float twoSigma2 = 2.0f * sigma * sigma;

        for (int dy = -filterRadius; dy <= filterRadius; dy++) {
            int y = clamp(int(tid.y) + dy, 0, int(kCausticRes) - 1);
            for (int dx = -filterRadius; dx <= filterRadius; dx++) {
                int x = clamp(int(tid.x) + dx, 0, int(kCausticRes) - 1);

                float dist2 = float(dx * dx + dy * dy);
                float weight = exp(-dist2 / twoSigma2);

                uint baseIdx = (uint(y) * kCausticRes + uint(x)) * 4u;
                float r = float(atomic_load_explicit(&causticBuffer[baseIdx + 0u], memory_order_relaxed)) * invFixedPoint;
                float g = float(atomic_load_explicit(&causticBuffer[baseIdx + 1u], memory_order_relaxed)) * invFixedPoint;
                float b = float(atomic_load_explicit(&causticBuffer[baseIdx + 2u], memory_order_relaxed)) * invFixedPoint;

                accumIrradiance += float3(r, g, b) * weight;
                filterWeightSum += weight;
            }
        }

        float3 finalCaustic = accumIrradiance / max(1e-5f, filterWeightSum);
        causticTexture.write(float4(finalCaustic, 1.0f), tid);
    }
}

kernel void renderSceneKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::write> outTexture [[texture(0)]],
    texture2d<float, access::sample> causticTexture [[texture(1)]],
    texture2d<float, access::sample> irradianceAtlas [[texture(2)]],
    constant GlassUniforms &uniforms [[buffer(0)]],
    device const GPUBVHNode *bvhNodes [[buffer(1)]],
    device const GPUTriangle *triangles [[buffer(2)]]
) {
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) return;

    float2 uv = (float2(tid) + 0.5f) / float2(uniforms.width, uniforms.height);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    float4 clipPos = float4(ndc.x, ndc.y, 1.0f, 1.0f);
    float4 viewPos = uniforms.projectionInverse * clipPos;
    viewPos /= viewPos.w;
    float3 worldDir = normalize((uniforms.viewInverse * float4(viewPos.xyz, 0.0f)).xyz);

    Ray primaryRay;
    primaryRay.origin = uniforms.cameraPosition;
    primaryRay.direction = worldDir;

    HitRecord primaryHit = intersectScene(primaryRay, true, bvhNodes, triangles, uniforms.numTeapotNodes);

    float3 pixelColor = float3(0.0f);

    if (!primaryHit.hit) {
        pixelColor = getSkyRadiance(primaryRay.direction, normalize(uniforms.sunDirection));
    } else if (!primaryHit.isGlass) {
        pixelColor = evaluateSurfaceRadiance(primaryHit, primaryRay, uniforms, uniforms.renderMode != 0, causticTexture, irradianceAtlas, bvhNodes, triangles);
    } else {
        float3 P1 = primaryHit.position;
        float3 N1 = primaryHit.normal;
        float3 V = -primaryRay.direction;
        float cosThetaI = clamp(dot(N1, V), 0.0f, 1.0f);

        float objIor = primaryHit.ior;
        float objDisp = primaryHit.dispersion;
        float3 objAbs = primaryHit.absorption;

        if (uniforms.renderMode == 0) {
            float F = dielectricFresnel(cosThetaI, 1.0f, objIor);

            float3 R = reflect(primaryRay.direction, N1);
            Ray reflRay;
            reflRay.origin = P1 + N1 * 0.002f;
            reflRay.direction = R;
            HitRecord reflHit = intersectScene(reflRay, false, bvhNodes, triangles, 0);
            float3 reflColor = evaluateSurfaceRadiance(reflHit, reflRay, uniforms, false, causticTexture, irradianceAtlas, bvhNodes, triangles);

            float3 T;
            float3 refrColor = float3(0.0f);
            if (refractRay(primaryRay.direction, N1, 1.0f / objIor, T)) {
                Ray insideRay;
                insideRay.origin = P1 - N1 * 0.002f;
                insideRay.direction = T;

                float tExit;
                float3 nExit;
                bool exitOk = false;
                if (primaryHit.objectId == 10) {
                    exitOk = intersectTeapotBVH(insideRay, bvhNodes, triangles, uniforms.numTeapotNodes, tExit, nExit);
                } else if (primaryHit.objectId == 11) {
                    exitOk = intersectSphere(insideRay, kSphereCenter, kSphereRadius, 0.001f, tExit, nExit);
                } else if (primaryHit.objectId == 12) {
                    exitOk = intersectCylinder(insideRay, kCylinderCenter, kCylinderRadius, kCylinderHeight, 0.001f, tExit, nExit);
                } else {
                    exitOk = intersectTriangularPrism(insideRay, kPrismCenter, kPrismSide, kPrismHeight, 0.001f, tExit, nExit);
                }

                if (exitOk) {
                    float3 P2 = insideRay.origin + insideRay.direction * tExit;
                    float3 N2 = -nExit;
                    float3 T2;
                    if (refractRay(T, N2, objIor / 1.0f, T2)) {
                        Ray exitRay;
                        exitRay.origin = P2 + T2 * 0.005f;
                        exitRay.direction = T2;
                        HitRecord exitHit = intersectScene(exitRay, false, bvhNodes, triangles, 0);
                        refrColor = evaluateSurfaceRadiance(exitHit, exitRay, uniforms, false, causticTexture, irradianceAtlas, bvhNodes, triangles);
                    }
                }
            }

            pixelColor = F * reflColor + (1.0f - F) * refrColor;
        } else {
            float disp = (uniforms.renderMode == 3) ? (objDisp * 1.6f) : objDisp;
            if ((uniforms.ablationMask & kAblDispersion) == 0u) disp = 0.0f;
            float iorR = objIor - disp;
            float iorG = objIor;
            float iorB = objIor + disp;

            float F_R = dielectricFresnel(cosThetaI, 1.0f, iorR);
            float F_G = dielectricFresnel(cosThetaI, 1.0f, iorG);
            float F_B = dielectricFresnel(cosThetaI, 1.0f, iorB);
            float3 F = float3(F_R, F_G, F_B);

            float3 reflDir = reflect(primaryRay.direction, N1);
            Ray reflRay;
            reflRay.origin = P1 + N1 * 0.003f;
            reflRay.direction = reflDir;
            HitRecord reflHit = intersectScene(reflRay, false, bvhNodes, triangles, 0);
            float3 reflColor = evaluateSurfaceRadiance(reflHit, reflRay, uniforms, true, causticTexture, irradianceAtlas, bvhNodes, triangles);

            float3 T_R, T_G, T_B;
            bool okR = refractRay(primaryRay.direction, N1, 1.0f / iorR, T_R);
            bool okG = refractRay(primaryRay.direction, N1, 1.0f / iorG, T_G);
            bool okB = refractRay(primaryRay.direction, N1, 1.0f / iorB, T_B);

            float3 refrColor = float3(0.0f);

            if (okG) {
                Ray insideRayG;
                insideRayG.origin = P1 + T_G * 0.003f;
                insideRayG.direction = T_G;

                float tExit;
                float3 nExit;
                bool exitOk = false;

                if (primaryHit.objectId == 10) {
                    exitOk = intersectTeapotBVH(insideRayG, bvhNodes, triangles, uniforms.numTeapotNodes, tExit, nExit);
                } else if (primaryHit.objectId == 11) {
                    exitOk = intersectSphere(insideRayG, kSphereCenter, kSphereRadius, 0.001f, tExit, nExit);
                } else if (primaryHit.objectId == 12) {
                    exitOk = intersectCylinder(insideRayG, kCylinderCenter, kCylinderRadius, kCylinderHeight, 0.001f, tExit, nExit);
                } else if (primaryHit.objectId == 13) {
                    exitOk = intersectTriangularPrism(insideRayG, kPrismCenter, kPrismSide, kPrismHeight, 0.001f, tExit, nExit);
                } else {
                    float3 slabMin = float3(1.00f, 0.0f, 0.35f);
                    float3 slabMax = float3(1.70f, 0.06f, 1.05f);
                    exitOk = intersectBox(insideRayG, slabMin, slabMax, 0.001f, tExit, nExit);
                }

                if (exitOk) {
                    float internalDist = tExit;
                    float3 P2 = insideRayG.origin + insideRayG.direction * tExit;
                    float3 N2 = -nExit;

                    float3 absorption = beerLambertAbsorption(objAbs, internalDist);

                    float3 T2_R, T2_G, T2_B;
                    bool exitOkR = refractRay(okR ? T_R : T_G, N2, iorR / 1.0f, T2_R);
                    bool exitOkG = refractRay(T_G, N2, iorG / 1.0f, T2_G);
                    bool exitOkB = refractRay(okB ? T_B : T_G, N2, iorB / 1.0f, T2_B);

                    if (uniforms.renderMode == 2) {
                        float coneAngle = uniforms.glassRoughness * 0.28f;
                        float3 wT = normalize(exitOkG ? T2_G : reflect(T_G, nExit));
                        float3 upVec = abs(wT.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                        float3 uVec = normalize(cross(wT, upVec));
                        float3 vVec = cross(wT, uVec);

                        // Per-pixel spiral rotation, otherwise the 16 taps line
                        // up across neighbours and the cone bands visibly.
                        float3 ignMagic = float3(0.06711056f, 0.00583715f, 52.9829189f);
                        float ign = fract(ignMagic.z * fract(dot(float2(tid), ignMagic.xy)));
                        float phi = ign * 6.283185307f;
                        float cosPhi = cos(phi);
                        float sinPhi = sin(phi);

                        constexpr int kSamples = 16;
                        float3 accumRad = float3(0.0f);
                        float weightSum = 0.0f;

                        for (int s = 0; s < kSamples; s++) {
                            float theta = float(s) * 2.39996323f;
                            float r = sqrt((float(s) + 0.5f) / float(kSamples));

                            float unrotX = r * cos(theta);
                            float unrotY = r * sin(theta);
                            float rotX = unrotX * cosPhi - unrotY * sinPhi;
                            float rotY = unrotX * sinPhi + unrotY * cosPhi;

                            float2 off = float2(rotX, rotY) * coneAngle;
                            float3 sampleDir = normalize(wT + off.x * uVec + off.y * vVec);

                            // Falls off towards the rim so the cone has no hard edge.
                            float weight = exp(-1.2f * r * r);

                            Ray coneRay;
                            coneRay.origin = P2 + sampleDir * 0.005f;
                            coneRay.direction = sampleDir;
                            HitRecord coneHit = intersectScene(coneRay, false, bvhNodes, triangles, 0);
                            accumRad += weight * evaluateSurfaceRadiance(coneHit, coneRay, uniforms, true, causticTexture, irradianceAtlas, bvhNodes, triangles);
                            weightSum += weight;
                        }
                        refrColor = (accumRad / weightSum) * absorption;
                    } else {
                        float3 dirR = exitOkR ? T2_R : reflect(okR ? T_R : T_G, nExit);
                        Ray exitRayR;
                        exitRayR.origin = P2 + dirR * 0.012f;
                        exitRayR.direction = dirR;
                        HitRecord hitR = intersectScene(exitRayR, false, bvhNodes, triangles, 0);
                        float radR = evaluateSurfaceRadiance(hitR, exitRayR, uniforms, true, causticTexture, irradianceAtlas, bvhNodes, triangles).r;

                        float3 dirG = exitOkG ? T2_G : reflect(T_G, nExit);
                        Ray exitRayG;
                        exitRayG.origin = P2 + dirG * 0.012f;
                        exitRayG.direction = dirG;
                        HitRecord hitG = intersectScene(exitRayG, false, bvhNodes, triangles, 0);
                        float radG = evaluateSurfaceRadiance(hitG, exitRayG, uniforms, true, causticTexture, irradianceAtlas, bvhNodes, triangles).g;

                        float3 dirB = exitOkB ? T2_B : reflect(okB ? T_B : T_G, nExit);
                        Ray exitRayB;
                        exitRayB.origin = P2 + dirB * 0.012f;
                        exitRayB.direction = dirB;
                        HitRecord hitB = intersectScene(exitRayB, false, bvhNodes, triangles, 0);
                        float radB = evaluateSurfaceRadiance(hitB, exitRayB, uniforms, true, causticTexture, irradianceAtlas, bvhNodes, triangles).b;

                        refrColor = float3(radR, radG, radB) * absorption;
                    }
                } else {
                    Ray escapeRay;
                    escapeRay.origin = P1 + T_G * 0.02f;
                    escapeRay.direction = T_G;
                    HitRecord escapeHit = intersectScene(escapeRay, false, bvhNodes, triangles, 0);
                    refrColor = evaluateSurfaceRadiance(escapeHit, escapeRay, uniforms, true, causticTexture, irradianceAtlas, bvhNodes, triangles);
                }
            }

            pixelColor = F * reflColor + (float3(1.0f) - F) * refrColor;

            float3 L = normalize(uniforms.sunDirection);
            float3 H = normalize(L + V);
            float NdotH = max(0.0f, dot(N1, H));
            float specPow = (uniforms.renderMode == 2) ? 24.0f : 512.0f;
            float specHighlight = pow(NdotH, specPow) * 1.5f;
            pixelColor += uniforms.sunColor * uniforms.sunIntensity * specHighlight * F;
        }
    }

    float3 finalColor = toneMapACES(pixelColor);
    outTexture.write(float4(finalColor, 1.0f), tid);
}

// ---------------------------------------------------------------------------
// Mode 4: brute-force path traced reference
//
// Same scene, same materials, same tone map as the real-time modes - the only
// thing that changes is that transport is solved by sampling paths instead of
// by the cascade + splat approximation. It exists to be the ground truth the
// other four modes are measured against, not to be fast.
//
// Conventions are matched to the raster path on purpose, so that a converged
// reference and a cascade frame are directly comparable:
//   * The sun is a disc of angular radius `sunAngularRadius` whose radiance is
//     pi * sunIntensity / omega, which reproduces exactly the raster path's
//     `albedo * sunIntensity * NdotL` for an unshadowed diffuse hit.
//   * The sky is the same getSkyRadiance() the cascades gather, so the ambient
//     level is the one the cascade pass is trying to reproduce.
//   * Glass is a smooth dielectric: Fresnel-weighted choice between one
//     reflection and one refraction, Beer-Lambert over the interior segment,
//     and a lazily picked spectral band for dispersion.
// ---------------------------------------------------------------------------

inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float randUniform(thread uint &state) {
    state = state * 1664525u + 1013904223u;
    return float(pcgHash(state) & 0x00FFFFFFu) / 16777216.0f;
}

inline float3 sampleCosineHemisphere(Basis tbn, float u1, float u2) {
    float r = sqrt(u1);
    float phi = 2.0f * kPi * u2;
    return tbn.toWorld(float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - u1))));
}

inline float3 sampleCone(float3 axis, float cosMax, float u1, float u2) {
    float cosTheta = 1.0f - u1 * (1.0f - cosMax);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * kPi * u2;
    Basis b = makeTBN(axis);
    return b.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));
}

// GGX half-vector, sampled from D (Walter et al. 2007). Mode 2's raster path
// approximates a rough interface by jittering the exit ray inside a cone of
// `roughness * 0.28` radians, so the reference uses the same number as alpha and
// roughens both interfaces properly instead of only the exit one.
inline float3 sampleGGXNormal(Basis tbn, float alpha, float u1, float u2) {
    float phi = 2.0f * kPi * u1;
    float a2 = alpha * alpha;
    float cosTheta = sqrt(max(0.0f, (1.0f - u2) / (1.0f + (a2 - 1.0f) * u2)));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    return tbn.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));
}

inline float smithG1(float3 v, float3 N, float3 H, float alpha) {
    float vn = dot(v, N);
    if (dot(v, H) * vn <= 0.0f) return 0.0f;
    float vn2 = vn * vn;
    float tan2 = (1.0f - vn2) / max(1e-6f, vn2);
    return 2.0f / (1.0f + sqrt(max(0.0f, 1.0f + alpha * alpha * tan2)));
}

// Anything the path escapes into. `includeSunDisc` is set only after a specular
// (or camera) bounce - a diffuse vertex already sampled the sun explicitly, so
// counting the disc again there would double the direct term.
inline float3 environmentRadiance(
    float3 dir,
    float3 sunDir,
    float3 sunRadiance,
    float cosSunMax,
    bool includeSunDisc
) {
    float3 L = getSkyRadiance(dir, sunDir);
    if (includeSunDisc && dot(dir, sunDir) >= cosSunMax) {
        L += sunRadiance;
    }
    return L;
}

kernel void pathTraceKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::read_write> accumTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant GlassUniforms &uniforms [[buffer(0)]],
    constant PathTraceParams &pt [[buffer(1)]],
    device const GPUBVHNode *bvhNodes [[buffer(2)]],
    device const GPUTriangle *triangles [[buffer(3)]]
) {
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) return;

    float3 sunDir = normalize(uniforms.sunDirection);
    float cosSunMax = cos(max(1e-4f, pt.sunAngularRadius));
    float sunSolidAngle = 2.0f * kPi * (1.0f - cosSunMax);
    // pi / omega, so that the reference's unshadowed direct term equals the
    // raster path's sunColor * sunIntensity * NdotL exactly.
    float3 sunRadiance = uniforms.sunColor * (uniforms.sunIntensity * kPi / sunSolidAngle);

    uint rngState = pcgHash(tid.x + tid.y * uniforms.width) ^
                    pcgHash((pt.sampleBase + pt.seedOffset) * 9781u + 1u);

    float3 batch = float3(0.0f);
    // Primary-hit id of the first path, parked in the accumulator's alpha so the
    // offline comparison can restrict its metrics to the floor or to the glass
    // without needing a separate G-buffer pass.
    float primaryObjectId = 0.0f;

    for (uint sampleIdx = 0u; sampleIdx < pt.samplesPerLaunch; sampleIdx++) {
        float2 jitter = float2(randUniform(rngState), randUniform(rngState));
        float2 uv = (float2(tid) + jitter) / float2(uniforms.width, uniforms.height);
        float2 ndc = uv * 2.0f - 1.0f;
        ndc.y = -ndc.y;

        float4 viewPos = uniforms.projectionInverse * float4(ndc.x, ndc.y, 1.0f, 1.0f);
        viewPos /= viewPos.w;

        Ray ray;
        ray.origin = uniforms.cameraPosition;
        ray.direction = normalize((uniforms.viewInverse * float4(viewPos.xyz, 0.0f)).xyz);

        float3 radiance = float3(0.0f);
        float3 throughput = float3(1.0f);

        bool includeSunDisc = true;   // the camera ray counts as a specular bounce
        bool insideGlass = false;
        float3 mediumAbsorption = float3(0.0f);

        // Dispersion is resolved lazily: a path stays achromatic until it first
        // meets a dispersive interface, then commits to one of three bands and
        // pays the 3x weight. Paths that never touch glass keep full colour.
        bool spectral = false;
        float bandOffset = 0.0f;
        float3 bandMask = float3(1.0f);

        for (uint depth = 0u; depth < pt.maxDepth; depth++) {
            HitRecord hit = intersectScene(ray, true, bvhNodes, triangles, uniforms.numTeapotNodes);

            if (sampleIdx == 0u && depth == 0u) {
                primaryObjectId = hit.hit ? float(hit.objectId) : 0.0f;
            }

            if (insideGlass && hit.hit) {
                throughput *= beerLambertAbsorption(mediumAbsorption, hit.distance);
            }

            if (!hit.hit) {
                radiance += throughput * environmentRadiance(ray.direction, sunDir, sunRadiance,
                                                             cosSunMax, includeSunDisc);
                break;
            }

            // Face the surface normal against the incoming ray. The mesh path
            // already flips its interpolated normal, the analytic primitives
            // return an outward one, so this normalises both cases; which side
            // of the interface we are on comes from `insideGlass`, not the sign.
            float3 N = dot(hit.normal, ray.direction) < 0.0f ? hit.normal : -hit.normal;

            if (hit.isGlass) {
                float objIor = hit.ior;
                float disp = (uniforms.renderMode == 3) ? (hit.dispersion * 1.6f) : hit.dispersion;
                if ((uniforms.ablationMask & kAblDispersion) == 0u) disp = 0.0f;

                if (!spectral && disp > 0.0f) {
                    float u = randUniform(rngState);
                    uint band = min(2u, uint(u * 3.0f));
                    bandOffset = (band == 0u) ? -1.0f : (band == 2u ? 1.0f : 0.0f);
                    bandMask = float3(band == 0u ? 1.0f : 0.0f,
                                      band == 1u ? 1.0f : 0.0f,
                                      band == 2u ? 1.0f : 0.0f);
                    throughput *= 3.0f * bandMask;
                    spectral = true;
                }
                float ior = objIor + bandOffset * disp;

                float iorI = insideGlass ? ior : 1.0f;
                float iorT = insideGlass ? 1.0f : ior;

                // Smooth by default; mode 2 turns the interface into a GGX
                // microfacet dielectric, which is what its cone hack stands in for.
                float alpha = (uniforms.renderMode == 2) ? (uniforms.glassRoughness * 0.28f) : 0.0f;
                float3 H = N;
                if (alpha > 1e-3f) {
                    H = sampleGGXNormal(makeTBN(N), alpha,
                                        randUniform(rngState), randUniform(rngState));
                    if (dot(H, -ray.direction) < 0.0f) H = -H;
                }

                float cosI = clamp(dot(-ray.direction, H), 0.0f, 1.0f);
                float F = dielectricFresnel(cosI, iorI, iorT);

                float3 nextDir;
                bool reflected = true;
                if (randUniform(rngState) >= F) {
                    // The Fresnel split is sampled exactly, so no weight applies.
                    if (refractRay(ray.direction, H, iorI / iorT, nextDir)) {
                        reflected = false;
                    }
                }
                if (reflected) {
                    nextDir = reflect(ray.direction, H);
                    if (dot(nextDir, N) <= 0.0f) break;   // microfacet self-shadowed
                } else {
                    insideGlass = !insideGlass;
                    mediumAbsorption = insideGlass ? hit.absorption : float3(0.0f);
                }

                if (alpha > 1e-3f) {
                    // D-sampled half vector, so the estimator keeps the Smith
                    // masking-shadowing ratio (Walter et al. 2007, eq. 38/41).
                    float G1o = smithG1(-ray.direction, N, H, alpha);
                    float G1i = smithG1(nextDir, N, H, alpha);
                    float denom = abs(dot(-ray.direction, N)) * abs(dot(N, H));
                    float weight = (denom > 1e-6f)
                                 ? abs(dot(-ray.direction, H)) * G1o * G1i / denom
                                 : 0.0f;
                    throughput *= min(weight, 4.0f);
                    if (weight <= 0.0f) break;
                }

                ray.origin = hit.position + nextDir * 0.0015f;
                ray.direction = nextDir;
                includeSunDisc = true;
                continue;
            }

            // Diffuse surface: explicit sun-disc sample, then a cosine bounce.
            Basis tbn = makeTBN(N);

            float3 wi = sampleCone(sunDir, cosSunMax,
                                   randUniform(rngState), randUniform(rngState));
            float NdotL = dot(N, wi);
            if (NdotL > 0.0f) {
                Ray shadowRay;
                shadowRay.origin = hit.position + N * 0.002f;
                shadowRay.direction = wi;
                // Glass occludes here, unlike the raster path's shadow ray. Light
                // that gets through arrives as a refracted specular path instead,
                // which is what makes the reference's caustics converge slowly.
                HitRecord occluder = intersectScene(shadowRay, true, bvhNodes, triangles, uniforms.numTeapotNodes);
                if (!occluder.hit) {
                    float3 direct = throughput * hit.albedo * sunRadiance *
                                    (NdotL * sunSolidAngle / kPi);
                    if (spectral) direct *= bandMask;
                    radiance += direct;
                }
            }

            // Cosine-weighted bounce: f * cos / pdf collapses to the albedo.
            throughput *= hit.albedo;
            float3 nextDir = sampleCosineHemisphere(tbn, randUniform(rngState), randUniform(rngState));
            ray.origin = hit.position + N * 0.002f;
            ray.direction = nextDir;
            includeSunDisc = false;

            if (depth >= pt.rrStartDepth) {
                float q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
                if (randUniform(rngState) > q) break;
                throughput /= q;
            }
        }

        if (pt.indirectClamp > 0.0f) {
            radiance = min(radiance, float3(pt.indirectClamp));
        }
        batch += radiance;
    }

    float4 prev = (pt.sampleBase == 0u) ? float4(0.0f) : accumTexture.read(tid);
    float objectId = (pt.sampleBase == 0u) ? primaryObjectId : prev.w;
    float4 accum = float4(prev.xyz + batch, objectId);
    accumTexture.write(accum, tid);

    float invSamples = 1.0f / float(pt.sampleBase + pt.samplesPerLaunch);
    outTexture.write(float4(toneMapACES(accum.xyz * invSamples * pt.exposure), 1.0f), tid);
}
