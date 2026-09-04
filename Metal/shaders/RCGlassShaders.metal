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
    float2 pad;
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

inline bool intersectBox(Ray ray, float3 bmin, float3 bmax, thread float &tHit, thread float3 &hitNormal) {
    float3 invD = 1.0f / (ray.direction + float3(1e-12f));
    float3 t0 = (bmin - ray.origin) * invD;
    float3 t1 = (bmax - ray.origin) * invD;

    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);

    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);

    if (enter > exit || exit < 0.001f) return false;

    float t = enter > 0.001f ? enter : exit;
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

inline bool intersectSphere(Ray ray, float3 center, float radius, thread float &tHit, thread float3 &hitNormal) {
    float3 oc = ray.origin - center;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return false;

    float sqrtDisc = sqrt(disc);
    float t0 = -b - sqrtDisc;
    float t1 = -b + sqrtDisc;

    if (t0 > 0.001f) {
        tHit = t0;
        hitNormal = normalize((ray.origin + ray.direction * t0) - center);
        return true;
    }
    if (t1 > 0.001f) {
        tHit = t1;
        hitNormal = normalize((ray.origin + ray.direction * t1) - center);
        return true;
    }
    return false;
}

inline bool intersectCylinder(Ray ray, float3 base, float radius, float height, thread float &tHit, thread float3 &hitNormal) {
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

            if (t0 > 0.001f) {
                float y = o.y + d.y * t0;
                if (y >= 0.0f && y <= height) {
                    tClosest = t0;
                    bestNorm = normalize(float3(o.x + d.x * t0, 0.0f, o.z + d.z * t0));
                    found = true;
                }
            }
            if (!found && t1 > 0.001f) {
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
        if (tCap > 0.001f && tCap < tClosest) {
            float x = o.x + d.x * tCap;
            float z = o.z + d.z * tCap;
            if (x * x + z * z <= radius * radius) {
                tClosest = tCap;
                bestNorm = float3(0.0f, 1.0f, 0.0f);
                found = true;
            }
        }
        float tBase = -o.y / d.y;
        if (tBase > 0.001f && tBase < tClosest) {
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

inline bool intersectTriangularPrism(Ray ray, float3 baseCenter, float side, float height, thread float &tHit, thread float3 &hitNormal) {
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
            if (t > 0.001f && t < tClosest) {
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
        if (tTop > 0.001f && tTop < tClosest) {
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
        if (tBot > 0.001f && tBot < tClosest) {
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

// Linear GPU BVH traversal using a fixed 64-depth stack with near-child sorting and interval bounds [tMin, tMax]
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

    // Linear BVH traversal using a fixed 64-depth stack
    int stack[64];
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

            if (hitL && hitR) {
                if (tNearL < tNearR) {
                    stack[stackPtr++] = node.rightChild;
                    stack[stackPtr++] = node.leftChild;
                } else {
                    stack[stackPtr++] = node.leftChild;
                    stack[stackPtr++] = node.rightChild;
                }
            } else if (hitL) {
                stack[stackPtr++] = node.leftChild;
            } else if (hitR) {
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

    // Room boundaries
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

        if (intersectSphere(ray, kSphereCenter, kSphereRadius, t, norm)) {
            if (t >= tMin && t < hit.distance) {
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

        if (intersectCylinder(ray, kCylinderCenter, kCylinderRadius, kCylinderHeight, t, norm)) {
            if (t >= tMin && t < hit.distance) {
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

        if (intersectTriangularPrism(ray, kPrismCenter, kPrismSide, kPrismHeight, t, norm)) {
            if (t >= tMin && t < hit.distance) {
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
        if (intersectBox(ray, slabMin, slabMax, t, norm)) {
            if (t >= tMin && t < hit.distance) {
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

inline float3 evaluateBRDF_GGX(float3 w_o, float3 w_i, float3 n, float alpha, float3 F0) {
    float3 h = normalize(w_i + w_o);
    float a2 = max(1e-4f, alpha * alpha);
    float NdotH = max(0.0f, dot(n, h));
    float denomD = (NdotH * NdotH * (a2 - 1.0f) + 1.0f);
    float D = a2 / (kPi * denomD * denomD);

    float NdotV = max(1e-4f, dot(n, w_o));
    float NdotL = max(1e-4f, dot(n, w_i));

    float3 F = F0 + (float3(1.0f) - F0) * pow(saturate(1.0f - NdotV), 5.0f);

    float k = a2 * 0.5f;
    float G = 1.0f / ((NdotL * (1.0f - k) + k) * (NdotV * (1.0f - k) + k));

    return F * (D * G * 0.25f);
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

// =========================================================================
// 3D Radiance Cascades Implementation (Alexander Sannikov)
// 4 Cascades with bounded geometric range intervals and hierarchical merging
// =========================================================================
constant float kCascadeRanges[5] = { 0.005f, 0.25f, 0.80f, 2.50f, 100.0f };

constant int kRaysC0 = 16;
constant int kRaysC1 = 32;
constant int kRaysC2 = 64;
constant int kRaysC3 = 128;

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

    // 3D objects in the room (Teapot, sphere, cylinder, prism)
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
        return float4(hitRad, boundaryFade); // w is residual transmittance
    }
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

kernel void computeRadianceCascadesKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::write> irradianceAtlas [[texture(0)]],
    texture2d<float, access::sample> prevAtlas [[texture(1)]],
    constant GlassUniforms &uniforms [[buffer(0)]],
    device const GPUBVHNode *bvhNodes [[buffer(1)]],
    device const GPUTriangle *triangles [[buffer(2)]]
) {
    if (tid.x >= kAtlasSurfaceWidth * kNumSurfaces || tid.y >= kAtlasSurfaceHeight) return;

    uint surfaceId = tid.x / kAtlasSurfaceWidth;
    uint lx = tid.x % kAtlasSurfaceWidth;
    uint ly = tid.y;

    float3 sunDir = normalize(uniforms.sunDirection);
    float jitter = fract(sin(dot(float2(lx, ly) + float2(surfaceId * 37.0f), float2(12.9898f, 78.233f))) * 43758.5453f) * (2.0f * kPi);
    const float kSurfaceRes = float(kAtlasSurfaceWidth);

    // ---------------------------------------------------------------------
    // 1. Cascade 3 (Far field: [2.50, 100.0] meters, 128 directions)
    // Coarse spatial probe grid (stride 8 -> 8x8 probes per surface)
    // ---------------------------------------------------------------------
    int bx3 = clamp(int(lx) / 8 * 8, 0, int(kAtlasSurfaceWidth) - 8);
    int by3 = clamp(int(ly) / 8 * 8, 0, int(kAtlasSurfaceHeight) - 8);
    float2 uvC3 = (float2(bx3 + 4, by3 + 4) + 0.5f) / kSurfaceRes;
    float3 probePosC3, probeNorC3;
    getSurfaceGeometry(surfaceId, uvC3, probePosC3, probeNorC3);
    Basis tbnC3 = makeTBN(probeNorC3);
    float3 rayOriginC3 = probePosC3 + probeNorC3 * 0.004f;

    float3 c3_rad[128];
    for (int i = 0; i < kRaysC3; i++) {
        float cosTheta = sqrt(max(0.0f, 1.0f - (float(i) + 0.5f) / float(kRaysC3)));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = float(i) * 2.399963229728f + jitter;
        float3 dir = tbnC3.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));

        float4 res = traceCascadeInterval(rayOriginC3, dir, kCascadeRanges[3], kCascadeRanges[4],
                                          sunDir, uniforms.sunColor, uniforms.sunIntensity, uniforms.numTeapotNodes,
                                          bvhNodes, triangles, prevAtlas);
        if (res.w > 0.0f) {
            c3_rad[i] = res.xyz + res.w * getSkyRadiance(dir, sunDir);
        } else {
            c3_rad[i] = res.xyz;
        }
    }

    // ---------------------------------------------------------------------
    // 2. Cascade 2 (Mid-to-far field: [0.80, 2.50] meters, 64 directions)
    // Mid spatial probe grid (stride 4 -> 16x16 probes per surface)
    // Hierarchically merges radiance from Cascade 3
    // ---------------------------------------------------------------------
    int bx2 = clamp(int(lx) / 4 * 4, 0, int(kAtlasSurfaceWidth) - 4);
    int by2 = clamp(int(ly) / 4 * 4, 0, int(kAtlasSurfaceHeight) - 4);
    float2 uvC2 = (float2(bx2 + 2, by2 + 2) + 0.5f) / kSurfaceRes;
    float3 probePosC2, probeNorC2;
    getSurfaceGeometry(surfaceId, uvC2, probePosC2, probeNorC2);
    Basis tbnC2 = makeTBN(probeNorC2);
    float3 rayOriginC2 = probePosC2 + probeNorC2 * 0.004f;

    float3 c2_rad[64];
    for (int i = 0; i < kRaysC2; i++) {
        float cosTheta = sqrt(max(0.0f, 1.0f - (float(i) + 0.5f) / float(kRaysC2)));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = float(i) * 2.399963229728f + jitter;
        float3 dir = tbnC2.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));

        float4 res = traceCascadeInterval(rayOriginC2, dir, kCascadeRanges[2], kCascadeRanges[3],
                                          sunDir, uniforms.sunColor, uniforms.sunIntensity, uniforms.numTeapotNodes,
                                          bvhNodes, triangles, prevAtlas);
        float3 incomingC3 = 0.5f * (c3_rad[2 * i] + c3_rad[2 * i + 1]);
        c2_rad[i] = res.xyz + res.w * incomingC3;
    }

    // ---------------------------------------------------------------------
    // 3. Cascade 1 (Near-to-mid field: [0.25, 0.80] meters, 32 directions)
    // Fine-mid spatial probe grid (stride 2 -> 32x32 probes per surface)
    // Hierarchically merges radiance from Cascade 2
    // ---------------------------------------------------------------------
    int bx1 = clamp(int(lx) / 2 * 2, 0, int(kAtlasSurfaceWidth) - 2);
    int by1 = clamp(int(ly) / 2 * 2, 0, int(kAtlasSurfaceHeight) - 2);
    float2 uvC1 = (float2(bx1 + 1, by1 + 1) + 0.5f) / kSurfaceRes;
    float3 probePosC1, probeNorC1;
    getSurfaceGeometry(surfaceId, uvC1, probePosC1, probeNorC1);
    Basis tbnC1 = makeTBN(probeNorC1);
    float3 rayOriginC1 = probePosC1 + probeNorC1 * 0.004f;

    float3 c1_rad[32];
    for (int i = 0; i < kRaysC1; i++) {
        float cosTheta = sqrt(max(0.0f, 1.0f - (float(i) + 0.5f) / float(kRaysC1)));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = float(i) * 2.399963229728f + jitter;
        float3 dir = tbnC1.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));

        float4 res = traceCascadeInterval(rayOriginC1, dir, kCascadeRanges[1], kCascadeRanges[2],
                                          sunDir, uniforms.sunColor, uniforms.sunIntensity, uniforms.numTeapotNodes,
                                          bvhNodes, triangles, prevAtlas);
        float3 incomingC2 = 0.5f * (c2_rad[2 * i] + c2_rad[2 * i + 1]);
        c1_rad[i] = res.xyz + res.w * incomingC2;
    }

    // ---------------------------------------------------------------------
    // 4. Cascade 0 (Contact field: [0.005, 0.25] meters, 16 directions)
    // Full resolution probe grid (stride 1 -> 64x64 probes per surface)
    // Hierarchically merges radiance from Cascade 1 & integrates irradiance
    // ---------------------------------------------------------------------
    float2 uvC0 = (float2(lx, ly) + 0.5f) / kSurfaceRes;
    float3 probePosC0, probeNorC0;
    getSurfaceGeometry(surfaceId, uvC0, probePosC0, probeNorC0);
    Basis tbnC0 = makeTBN(probeNorC0);
    float3 rayOriginC0 = probePosC0 + probeNorC0 * 0.004f;

    float3 c0_rad[16];
    float3 accumIrradiance = float3(0.0f);

    for (int i = 0; i < kRaysC0; i++) {
        float cosTheta = sqrt(max(0.0f, 1.0f - (float(i) + 0.5f) / float(kRaysC0)));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = float(i) * 2.399963229728f + jitter;
        float3 dir = tbnC0.toWorld(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));

        float4 res = traceCascadeInterval(rayOriginC0, dir, kCascadeRanges[0], kCascadeRanges[1],
                                          sunDir, uniforms.sunColor, uniforms.sunIntensity, uniforms.numTeapotNodes,
                                          bvhNodes, triangles, prevAtlas);
        float3 incomingC1 = 0.5f * (c1_rad[2 * i] + c1_rad[2 * i + 1]);
        c0_rad[i] = res.xyz + res.w * incomingC1;

        accumIrradiance += c0_rad[i];
    }

    float3 newIrradiance = accumIrradiance / float(kRaysC0);
    float3 prevVal = prevAtlas.read(tid).rgb;
    float3 finalIrradiance = (uniforms.frameIndex > 1) ? mix(newIrradiance, prevVal, 0.70f) : newIrradiance;
    irradianceAtlas.write(float4(finalIrradiance, 1.0f), tid);
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
    if (hit.objectId == 2) {
        float uFloor = (hit.position.x - kFloorMinX) / (kFloorMaxX - kFloorMinX);
        float vFloor = (hit.position.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ);
        if (uFloor >= 0.0f && uFloor <= 1.0f && vFloor >= 0.0f && vFloor <= 1.0f) {
            constexpr sampler causticSampler(coord::normalized, filter::linear, address::clamp_to_edge);
            causticRad = causticTexture.sample(causticSampler, float2(uFloor, vFloor)).rgb;
        }
    }

    float3 indirectGI = float3(0.0f);
    if (useCascadeGI) {
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

kernel void generateCausticsKernel(
    uint2 tid [[thread_position_in_grid]],
    device atomic_uint *causticBuffer [[buffer(0)]],
    constant GlassUniforms &uniforms [[buffer(1)]],
    device const GPUBVHNode *bvhNodes [[buffer(2)]],
    device const GPUTriangle *triangles [[buffer(3)]]
) {
    if (tid.x >= 2048 || tid.y >= 2048) return;

    uint qX = tid.x / 1024;
    uint qY = tid.y / 1024;
    uint quadrant = qX + qY * 2;

    uint lx = tid.x % 1024;
    uint ly = tid.y % 1024;
    float2 uv = (float2(lx, ly) + 0.5f) / 1024.0f;

    float3 L = normalize(uniforms.sunDirection);
    float3 lightDir = -L;

    float3 up = abs(L.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 uAxis = normalize(cross(L, up));
    float3 vAxis = cross(L, uAxis);

    const float kFixedScale = 1000000000.0f;

    // Quadrant 0: Sphere analytical caustics
    if (quadrant == 0) {
        float sx = (uv.x - 0.5f) * (2.0f * kSphereRadius);
        float sy = (uv.y - 0.5f) * (2.0f * kSphereRadius);
        float s2 = sx * sx + sy * sy;
        if (s2 >= kSphereRadius * kSphereRadius) return;

        float sz = sqrt(max(0.0f, kSphereRadius * kSphereRadius - s2));
        float3 P1 = kSphereCenter + sx * uAxis + sy * vAxis + sz * L;
        float3 N1 = (P1 - kSphereCenter) / kSphereRadius;

        float cosTheta1 = clamp(-dot(lightDir, N1), 0.0f, 1.0f);
        float rayWeight = (4.0f * kSphereRadius * kSphereRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        float disp = (uniforms.renderMode == 3) ? (0.040f * 1.5f) : 0.025f;
        float iorR = 1.62f - disp;
        float iorG = 1.62f;
        float iorB = 1.62f + disp;
        float3 iors = float3(iorR, iorG, iorB);

        for (int ch = 0; ch < 3; ch++) {
            float eta = iors[ch];

            float sin2Theta2 = (1.0f - cosTheta1 * cosTheta1) / (eta * eta);
            if (sin2Theta2 >= 1.0f) continue;
            float cosTheta2 = sqrt(1.0f - sin2Theta2);

            float3 D1 = (lightDir / eta) + (cosTheta1 / eta - cosTheta2) * N1;
            float T_entry = 1.0f - dielectricFresnel(cosTheta1, 1.0f, eta);

            float internalDist = 2.0f * kSphereRadius * cosTheta2;
            float3 P2 = P1 + D1 * internalDist;
            float3 N2 = (P2 - kSphereCenter) / kSphereRadius;

            float absorption = exp(-0.02f * internalDist);

            float cosTheta3 = clamp(dot(D1, N2), 0.0f, 1.0f);
            float sin2Theta4 = (1.0f - cosTheta3 * cosTheta3) * (eta * eta);
            if (sin2Theta4 >= 1.0f) continue;
            float cosTheta4 = sqrt(1.0f - sin2Theta4);

            float3 D2 = eta * D1 - (eta * cosTheta3 - cosTheta4) * N2;
            float T_exit = 1.0f - dielectricFresnel(cosTheta3, eta, 1.0f);

            float flux = rayWeight * T_entry * T_exit * absorption * uniforms.sunColor[ch];

            if (D2.y < -1e-4f) {
                float tFloor = -P2.y / D2.y;
                if (tFloor > 0.0f) {
                    float3 hitFloor = P2 + D2 * tFloor;
                    if (hitFloor.x >= kFloorMinX && hitFloor.x <= kFloorMaxX &&
                        hitFloor.z >= kFloorMinZ && hitFloor.z <= kFloorMaxZ) {

                        float uNorm = (hitFloor.x - kFloorMinX) / (kFloorMaxX - kFloorMinX);
                        float vNorm = (hitFloor.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ);

                        float gx = uNorm * float(kCausticRes) - 0.5f;
                        float gy = vNorm * float(kCausticRes) - 0.5f;

                        int x0 = int(floor(gx));
                        int y0 = int(floor(gy));
                        float fx = gx - float(x0);
                        float fy = gy - float(y0);

                        float w00 = (1.0f - fx) * (1.0f - fy);
                        float w10 = fx * (1.0f - fy);
                        float w01 = (1.0f - fx) * fy;
                        float w11 = fx * fy;

                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w00 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w10 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w01 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w11 * kFixedScale), memory_order_relaxed);
                        }
                    }
                }
            }
        }
        return;
    }

    // Quadrant 1: Utah Teapot mesh caustics via BVH
    if (quadrant == 1) {
        if (uniforms.numTeapotNodes == 0) return;
        float objRadius = 0.48f;
        float sx = (uv.x - 0.5f) * (2.0f * objRadius);
        float sy = (uv.y - 0.5f) * (2.0f * objRadius);
        if (sx * sx + sy * sy > objRadius * objRadius) return;

        float3 rayOrigin = float3(0.0f, 0.30f, 0.0f) + sx * uAxis + sy * vAxis - lightDir * 3.5f;
        Ray photonRay;
        photonRay.origin = rayOrigin;
        photonRay.direction = lightDir;

        float tEntry;
        float3 nEntry;
        if (!intersectTeapotBVH(photonRay, bvhNodes, triangles, uniforms.numTeapotNodes, tEntry, nEntry)) return;

        float3 P1 = photonRay.origin + photonRay.direction * tEntry;
        float3 N1 = nEntry;
        float cosTheta1 = clamp(-dot(lightDir, N1), 0.0f, 1.0f);

        float rayWeight = (4.0f * objRadius * objRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        float objIor = 1.52f;
        float3 D1;
        if (!refractRay(lightDir, N1, 1.0f / objIor, D1)) return;
        float T_entry = 1.0f - dielectricFresnel(cosTheta1, 1.0f, objIor);

        Ray insideRay;
        insideRay.origin = P1 + D1 * 0.005f;
        insideRay.direction = D1;

        float tExit;
        float3 nExit;
        if (!intersectTeapotBVH(insideRay, bvhNodes, triangles, uniforms.numTeapotNodes, tExit, nExit)) return;
        if (tExit < 0.002f) return;

        float internalDist = tExit;
        float3 P2 = insideRay.origin + insideRay.direction * tExit;
        float3 N2 = -nExit;
        float cosTheta3 = clamp(dot(D1, N2), 0.0f, 1.0f);
        float T_exit = 1.0f - dielectricFresnel(cosTheta3, objIor, 1.0f);

        float3 absorption = beerLambertAbsorption(float3(0.05f), internalDist);
        float3 baseFlux = rayWeight * T_entry * T_exit * absorption * uniforms.sunColor;

        float disp = (uniforms.renderMode == 3) ? (0.025f * 1.5f) : 0.0f;
        float iors[3] = { objIor - disp, objIor, objIor + disp };

        for (int ch = 0; ch < 3; ch++) {
            float eta = iors[ch];
            float sin2Theta4 = (1.0f - cosTheta3 * cosTheta3) * (eta * eta);
            if (sin2Theta4 >= 1.0f) continue;
            float cosTheta4 = sqrt(1.0f - sin2Theta4);
            float3 D2 = eta * D1 - (eta * cosTheta3 - cosTheta4) * N2;

            if (D2.y < -1e-4f) {
                float tFloor = -P2.y / D2.y;
                if (tFloor > 0.0f) {
                    float3 hitFloor = P2 + D2 * tFloor;
                    if (hitFloor.x >= kFloorMinX && hitFloor.x <= kFloorMaxX &&
                        hitFloor.z >= kFloorMinZ && hitFloor.z <= kFloorMaxZ) {

                        float uNorm = (hitFloor.x - kFloorMinX) / (kFloorMaxX - kFloorMinX);
                        float vNorm = (hitFloor.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ);

                        float gx = uNorm * float(kCausticRes) - 0.5f;
                        float gy = vNorm * float(kCausticRes) - 0.5f;

                        int x0 = int(floor(gx));
                        int y0 = int(floor(gy));
                        float fx = gx - float(x0);
                        float fy = gy - float(y0);

                        float w00 = (1.0f - fx) * (1.0f - fy);
                        float w10 = fx * (1.0f - fy);
                        float w01 = (1.0f - fx) * fy;
                        float w11 = fx * fy;

                        float flux = baseFlux[ch];

                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w00 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w10 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w01 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w11 * kFixedScale), memory_order_relaxed);
                        }
                    }
                }
            }
        }
        return;
    }

    // Quadrant 2: Cylinder analytical caustics
    if (quadrant == 2) {
        float objRadius = kCylinderRadius * 1.15f;
        float sx = (uv.x - 0.5f) * (2.0f * objRadius);
        float sy = (uv.y - 0.5f) * (2.0f * objRadius);
        if (sx * sx + sy * sy > objRadius * objRadius) return;

        float3 cylCenter = kCylinderCenter + float3(0.0f, kCylinderHeight * 0.5f, 0.0f);
        float3 rayOrigin = cylCenter + sx * uAxis + sy * vAxis - lightDir * 3.5f;
        Ray photonRay;
        photonRay.origin = rayOrigin;
        photonRay.direction = lightDir;

        float tEntry;
        float3 nEntry;
        if (!intersectCylinder(photonRay, kCylinderCenter, kCylinderRadius, kCylinderHeight, tEntry, nEntry)) return;

        float3 P1 = photonRay.origin + photonRay.direction * tEntry;
        float3 N1 = nEntry;
        float cosTheta1 = clamp(-dot(lightDir, N1), 0.0f, 1.0f);

        float rayWeight = (4.0f * objRadius * objRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        float objIor = 1.50f;
        float3 D1;
        if (!refractRay(lightDir, N1, 1.0f / objIor, D1)) return;
        float T_entry = 1.0f - dielectricFresnel(cosTheta1, 1.0f, objIor);

        Ray insideRay;
        insideRay.origin = P1 + D1 * 0.005f;
        insideRay.direction = D1;

        float tExit;
        float3 nExit;
        if (!intersectCylinder(insideRay, kCylinderCenter, kCylinderRadius, kCylinderHeight, tExit, nExit)) return;
        if (tExit < 0.002f) return;

        float internalDist = tExit;
        float3 P2 = insideRay.origin + insideRay.direction * tExit;
        float3 N2 = -nExit;
        float cosTheta3 = clamp(dot(D1, N2), 0.0f, 1.0f);
        float T_exit = 1.0f - dielectricFresnel(cosTheta3, objIor, 1.0f);

        float3 absorption = beerLambertAbsorption(float3(1.2f, 0.15f, 0.9f) * 2.2f, internalDist);
        float3 baseFlux = rayWeight * T_entry * T_exit * absorption * uniforms.sunColor;

        float sin2Theta4 = (1.0f - cosTheta3 * cosTheta3) * (objIor * objIor);
        if (sin2Theta4 >= 1.0f) return;
        float cosTheta4 = sqrt(1.0f - sin2Theta4);
        float3 D2 = objIor * D1 - (objIor * cosTheta3 - cosTheta4) * N2;

        if (D2.y < -1e-4f) {
            float tFloor = -P2.y / D2.y;
            if (tFloor > 0.0f) {
                float3 hitFloor = P2 + D2 * tFloor;
                if (hitFloor.x >= kFloorMinX && hitFloor.x <= kFloorMaxX &&
                    hitFloor.z >= kFloorMinZ && hitFloor.z <= kFloorMaxZ) {

                    float uNorm = (hitFloor.x - kFloorMinX) / (kFloorMaxX - kFloorMinX);
                    float vNorm = (hitFloor.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ);

                    float gx = uNorm * float(kCausticRes) - 0.5f;
                    float gy = vNorm * float(kCausticRes) - 0.5f;

                    int x0 = int(floor(gx));
                    int y0 = int(floor(gy));
                    float fx = gx - float(x0);
                    float fy = gy - float(y0);

                    float w00 = (1.0f - fx) * (1.0f - fy);
                    float w10 = fx * (1.0f - fy);
                    float w01 = (1.0f - fx) * fy;
                    float w11 = fx * fy;

                    for (int ch = 0; ch < 3; ch++) {
                        float flux = baseFlux[ch];
                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w00 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w10 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w01 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w11 * kFixedScale), memory_order_relaxed);
                        }
                    }
                }
            }
        }
        return;
    }

    // Quadrant 3: Triangular prism dispersion caustics
    {
        float objRadius = kPrismSide * 0.75f;
        float sx = (uv.x - 0.5f) * (2.0f * objRadius);
        float sy = (uv.y - 0.5f) * (2.0f * objRadius);
        if (sx * sx + sy * sy > objRadius * objRadius) return;

        float3 prismCenter = kPrismCenter + float3(0.0f, kPrismHeight * 0.5f, 0.0f);
        float3 rayOrigin = prismCenter + sx * uAxis + sy * vAxis - lightDir * 3.5f;
        Ray photonRay;
        photonRay.origin = rayOrigin;
        photonRay.direction = lightDir;

        float tEntry;
        float3 nEntry;
        if (!intersectTriangularPrism(photonRay, kPrismCenter, kPrismSide, kPrismHeight, tEntry, nEntry)) return;

        float3 P1 = photonRay.origin + photonRay.direction * tEntry;
        float3 N1 = nEntry;
        float cosTheta1 = clamp(-dot(lightDir, N1), 0.0f, 1.0f);

        float rayWeight = (4.0f * objRadius * objRadius / float(1024 * 1024)) * uniforms.sunIntensity;

        float baseIor = 1.58f;
        float prismDisp = 0.055f;
        float iors[3] = { baseIor - prismDisp, baseIor, baseIor + prismDisp };

        for (int ch = 0; ch < 3; ch++) {
            float eta = iors[ch];
            float3 D1;
            if (!refractRay(lightDir, N1, 1.0f / eta, D1)) continue;
            float T_entry = 1.0f - dielectricFresnel(cosTheta1, 1.0f, eta);

            Ray insideRay;
            insideRay.origin = P1 + D1 * 0.005f;
            insideRay.direction = D1;

            float tExit;
            float3 nExit;
            if (!intersectTriangularPrism(insideRay, kPrismCenter, kPrismSide, kPrismHeight, tExit, nExit)) continue;
            if (tExit < 0.002f) continue;

            float3 P2 = insideRay.origin + insideRay.direction * tExit;
            float3 N2 = -nExit;
            float cosTheta3 = clamp(dot(D1, N2), 0.0f, 1.0f);

            float sin2Theta4 = (1.0f - cosTheta3 * cosTheta3) * (eta * eta);
            if (sin2Theta4 >= 1.0f) continue;
            float cosTheta4 = sqrt(1.0f - sin2Theta4);
            float3 D2 = eta * D1 - (eta * cosTheta3 - cosTheta4) * N2;
            float T_exit = 1.0f - dielectricFresnel(cosTheta3, eta, 1.0f);

            float flux = rayWeight * T_entry * T_exit * uniforms.sunColor[ch] * 1.5f;

            if (D2.y < -1e-4f) {
                float tFloor = -P2.y / D2.y;
                if (tFloor > 0.0f) {
                    float3 hitFloor = P2 + D2 * tFloor;
                    if (hitFloor.x >= kFloorMinX && hitFloor.x <= kFloorMaxX &&
                        hitFloor.z >= kFloorMinZ && hitFloor.z <= kFloorMaxZ) {

                        float uNorm = (hitFloor.x - kFloorMinX) / (kFloorMaxX - kFloorMinX);
                        float vNorm = (hitFloor.z - kFloorMinZ) / (kFloorMaxZ - kFloorMinZ);

                        float gx = uNorm * float(kCausticRes) - 0.5f;
                        float gy = vNorm * float(kCausticRes) - 0.5f;

                        int x0 = int(floor(gx));
                        int y0 = int(floor(gy));
                        float fx = gx - float(x0);
                        float fy = gy - float(y0);

                        float w00 = (1.0f - fx) * (1.0f - fy);
                        float w10 = fx * (1.0f - fy);
                        float w01 = (1.0f - fx) * fy;
                        float w11 = fx * fy;

                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w00 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 >= 0 && y0 < int(kCausticRes)) {
                            uint idx = (uint(y0) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w10 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 >= 0 && x0 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w01 * kFixedScale), memory_order_relaxed);
                        }
                        if (x0 + 1 >= 0 && x0 + 1 < int(kCausticRes) && y0 + 1 >= 0 && y0 + 1 < int(kCausticRes)) {
                            uint idx = (uint(y0 + 1) * kCausticRes + uint(x0 + 1)) * 4u + uint(ch);
                            atomic_fetch_add_explicit(&causticBuffer[idx], uint(flux * w11 * kFixedScale), memory_order_relaxed);
                        }
                    }
                }
            }
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
    float invFixedPoint = 1.0f / (1000000000.0f * pixelArea);

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
                    exitOk = intersectSphere(insideRay, kSphereCenter, kSphereRadius, tExit, nExit);
                } else if (primaryHit.objectId == 12) {
                    exitOk = intersectCylinder(insideRay, kCylinderCenter, kCylinderRadius, kCylinderHeight, tExit, nExit);
                } else {
                    exitOk = intersectTriangularPrism(insideRay, kPrismCenter, kPrismSide, kPrismHeight, tExit, nExit);
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
                    exitOk = intersectSphere(insideRayG, kSphereCenter, kSphereRadius, tExit, nExit);
                } else if (primaryHit.objectId == 12) {
                    exitOk = intersectCylinder(insideRayG, kCylinderCenter, kCylinderRadius, kCylinderHeight, tExit, nExit);
                } else if (primaryHit.objectId == 13) {
                    exitOk = intersectTriangularPrism(insideRayG, kPrismCenter, kPrismSide, kPrismHeight, tExit, nExit);
                } else {
                    float3 slabMin = float3(1.00f, 0.0f, 0.35f);
                    float3 slabMax = float3(1.70f, 0.06f, 1.05f);
                    exitOk = intersectBox(insideRayG, slabMin, slabMax, tExit, nExit);
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

                        // High-frequency per-pixel rotation using Interleaved Gradient Noise (IGN)
                        float3 ignMagic = float3(0.06711056f, 0.00583715f, 52.9829189f);
                        float ign = fract(ignMagic.z * fract(dot(float2(tid), ignMagic.xy)));
                        float phi = ign * 6.283185307f;
                        float cosPhi = cos(phi);
                        float sinPhi = sin(phi);

                        // 16-sample Vogel's disk (golden ratio spiral) with Gaussian radial weighting
                        constexpr int kSamples = 16;
                        float3 accumRad = float3(0.0f);
                        float weightSum = 0.0f;

                        for (int s = 0; s < kSamples; s++) {
                            float theta = float(s) * 2.39996323f;
                            float r = sqrt((float(s) + 0.5f) / float(kSamples));

                            // Rotated disc offset
                            float unrotX = r * cos(theta);
                            float unrotY = r * sin(theta);
                            float rotX = unrotX * cosPhi - unrotY * sinPhi;
                            float rotY = unrotX * sinPhi + unrotY * cosPhi;

                            float2 off = float2(rotX, rotY) * coneAngle;
                            float3 sampleDir = normalize(wT + off.x * uVec + off.y * vVec);

                            // Smooth Gaussian-like weight to prevent harsh disc edges
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
