const float kPi = 3.14159265358979323846;
const uint kCausticRes = 1024u;
const float kFloorMinX = -2.5;
const float kFloorMaxX =  2.5;
const float kFloorMinZ = -2.5;
const float kFloorMaxZ =  2.5;

const vec3 kSphereCenter = vec3(1.10, 0.45, -0.25);
const float kSphereRadius = 0.45;

const vec3 kCylinderCenter = vec3(1.35, 0.06, 0.70);
const float kCylinderRadius = 0.28;
const float kCylinderHeight = 0.80;

const vec3 kPrismCenter = vec3(-1.25, 0.0, -0.20);
const float kPrismSide   = 0.55;
const float kPrismHeight = 0.70;

const float kRoomMinX = -2.5;
const float kRoomMaxX =  2.5;
const float kRoomMinY =  0.0;
const float kRoomMaxY =  3.5;
const float kRoomMinZ = -2.5;
const float kRoomMaxZ =  2.5;

const float kWinMinY = 0.35;
const float kWinMaxY = 3.25;
const float kWinMinZ = -2.10;
const float kWinMaxZ =  2.10;

const uint kAtlasSurfaceWidth = 64u;
const uint kAtlasSurfaceHeight = 64u;
const uint kNumSurfaces = 5u;

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct HitRecord {
    bool hit;
    float distance;
    vec3 position;
    vec3 normal;
    vec3 albedo;
    float roughness;
    bool isGlass;
    uint objectId;
    float ior;
    float dispersion;
    vec3 absorption;
};

struct GPUBVHNode {
    vec4 bmin;
    vec4 bmax;
    int leftChild;
    int rightChild;
    int pad0;
    int pad1;
};

struct GPUTriangle {
    vec4 v0;
    vec4 v1;
    vec4 v2;
    vec4 n0;
    vec4 n1;
    vec4 n2;
};

struct GlassUniformsData {
    mat4 viewInverse;
    mat4 projectionInverse;
    vec3 cameraPosition;
    float time;

    vec3 sunDirection;
    float sunIntensity;
    vec3 sunColor;
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
    vec2 pad;
};

struct Basis {
    vec3 tangent;
    vec3 bitangent;
    vec3 normal;
};

vec3 basisToWorld(Basis b, vec3 v) {
    return v.x * b.tangent + v.y * b.bitangent + v.z * b.normal;
}

Basis makeTBN(vec3 N) {
    Basis b;
    b.normal = N;
    if (abs(N.y) > 0.999) {
        b.tangent = vec3(1.0, 0.0, 0.0);
        b.bitangent = vec3(0.0, 0.0, 1.0);
    } else {
        b.tangent = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
        b.bitangent = cross(b.tangent, N);
    }
    return b;
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 saturate(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}

float dielectricFresnel(float cosThetaI, float iorI, float iorT) {
    cosThetaI = clamp(abs(cosThetaI), 0.0, 1.0);
    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = (iorI / iorT) * sinThetaI;
    if (sinThetaT >= 1.0) return 1.0;

    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    float rParallel = ((iorT * cosThetaI) - (iorI * cosThetaT)) /
                      ((iorT * cosThetaI) + (iorI * cosThetaT));
    float rPerp     = ((iorI * cosThetaI) - (iorT * cosThetaT)) /
                      ((iorI * cosThetaI) + (iorT * cosThetaT));
    return 0.5 * (rParallel * rParallel + rPerp * rPerp);
}

bool refractRay(vec3 I, vec3 N, float eta, out vec3 T) {
    if (dot(N, I) > 0.0) N = -N;
    float cosI = -dot(N, I);
    float sin2T = eta * eta * (1.0 - cosI * cosI);
    if (sin2T >= 1.0) {
        T = vec3(0.0);
        return false;
    }
    float cosT = sqrt(1.0 - sin2T);
    T = eta * I + (eta * cosI - cosT) * N;
    return true;
}

vec3 beerLambertAbsorption(vec3 absorptionCoeff, float dist) {
    return exp(-absorptionCoeff * dist);
}

vec3 toneMapACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

vec3 getSkyRadiance(vec3 direction, vec3 sunDir) {
    float sunDot = max(0.0, dot(direction, sunDir));
    vec3 zenithColor = vec3(0.40, 0.62, 0.95);
    vec3 horizonColor = vec3(0.78, 0.85, 0.95);
    vec3 sunGlowColor = vec3(1.0, 0.92, 0.75);

    float hFactor = saturate(direction.y * 1.5);
    vec3 sky = mix(horizonColor, zenithColor, hFactor);
    float sunDisc = (sunDot > 1e-3) ? (pow(sunDot, 128.0) * 4.0 + pow(sunDot, 1024.0) * 20.0) : 0.0;
    return sky + sunGlowColor * sunDisc;
}

bool intersectBox(Ray ray, vec3 bmin, vec3 bmax, out float tHit, out vec3 hitNormal) {
    vec3 invD = 1.0 / (ray.direction + vec3(1e-12));
    vec3 t0 = (bmin - ray.origin) * invD;
    vec3 t1 = (bmax - ray.origin) * invD;

    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);

    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);

    if (enter > exit || exit < 0.001) {
        tHit = 0.0;
        hitNormal = vec3(0.0);
        return false;
    }

    float t = enter > 0.001 ? enter : exit;
    tHit = t;

    vec3 p = ray.origin + ray.direction * t;
    vec3 center = 0.5 * (bmin + bmax);
    vec3 d = p - center;
    vec3 extent = 0.5 * (bmax - bmin);

    vec3 bias = d / extent;
    vec3 absBias = abs(bias);

    if (absBias.x > absBias.y && absBias.x > absBias.z) {
        hitNormal = vec3(sign(bias.x), 0.0, 0.0);
    } else if (absBias.y > absBias.z) {
        hitNormal = vec3(0.0, sign(bias.y), 0.0);
    } else {
        hitNormal = vec3(0.0, 0.0, sign(bias.z));
    }
    return true;
}

bool intersectBoxFast(Ray ray, vec3 bmin, vec3 bmax, out float tNear) {
    vec3 invD = 1.0 / (ray.direction + vec3(1e-12));
    vec3 t0 = (bmin - ray.origin) * invD;
    vec3 t1 = (bmax - ray.origin) * invD;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);
    tNear = enter;
    return (enter <= exit && exit > 0.001);
}

bool intersectSphere(Ray ray, vec3 center, float radius, out float tHit, out vec3 hitNormal) {
    vec3 oc = ray.origin - center;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        tHit = 0.0;
        hitNormal = vec3(0.0);
        return false;
    }

    float sqrtDisc = sqrt(disc);
    float t0 = -b - sqrtDisc;
    float t1 = -b + sqrtDisc;

    if (t0 > 0.001) {
        tHit = t0;
        hitNormal = normalize((ray.origin + ray.direction * t0) - center);
        return true;
    }
    if (t1 > 0.001) {
        tHit = t1;
        hitNormal = normalize((ray.origin + ray.direction * t1) - center);
        return true;
    }
    tHit = 0.0;
    hitNormal = vec3(0.0);
    return false;
}

bool intersectCylinder(Ray ray, vec3 base, float radius, float height, out float tHit, out vec3 hitNormal) {
    vec3 d = ray.direction;
    vec3 o = ray.origin - base;

    float a = d.x * d.x + d.z * d.z;
    float b = 2.0 * (o.x * d.x + o.z * d.z);
    float c = o.x * o.x + o.z * o.z - radius * radius;

    float tClosest = 1e30;
    vec3 bestNorm = vec3(0.0);
    bool found = false;

    if (a > 1e-6) {
        float disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            float sqrtD = sqrt(disc);
            float t0 = (-b - sqrtD) / (2.0 * a);
            float t1 = (-b + sqrtD) / (2.0 * a);

            if (t0 > 0.001) {
                float y = o.y + d.y * t0;
                if (y >= 0.0 && y <= height) {
                    tClosest = t0;
                    bestNorm = normalize(vec3(o.x + d.x * t0, 0.0, o.z + d.z * t0));
                    found = true;
                }
            }
            if (!found && t1 > 0.001) {
                float y = o.y + d.y * t1;
                if (y >= 0.0 && y <= height) {
                    tClosest = t1;
                    bestNorm = normalize(vec3(o.x + d.x * t1, 0.0, o.z + d.z * t1));
                    found = true;
                }
            }
        }
    }

    if (abs(d.y) > 1e-6) {
        float tCap = (height - o.y) / d.y;
        if (tCap > 0.001 && tCap < tClosest) {
            float x = o.x + d.x * tCap;
            float z = o.z + d.z * tCap;
            if (x * x + z * z <= radius * radius) {
                tClosest = tCap;
                bestNorm = vec3(0.0, 1.0, 0.0);
                found = true;
            }
        }
        float tBase = -o.y / d.y;
        if (tBase > 0.001 && tBase < tClosest) {
            float x = o.x + d.x * tBase;
            float z = o.z + d.z * tBase;
            if (x * x + z * z <= radius * radius) {
                tClosest = tBase;
                bestNorm = vec3(0.0, -1.0, 0.0);
                found = true;
            }
        }
    }

    if (found) {
        tHit = tClosest;
        hitNormal = bestNorm;
        return true;
    }
    tHit = 0.0;
    hitNormal = vec3(0.0);
    return false;
}

bool intersectTriangularPrism(Ray ray, vec3 baseCenter, float side, float height, out float tHit, out vec3 hitNormal) {
    float h = side * 0.8660254;
    vec3 p0 = baseCenter + vec3(0.0, 0.0, 2.0 * h / 3.0);
    vec3 p1 = baseCenter + vec3(-side * 0.5, 0.0, -h / 3.0);
    vec3 p2 = baseCenter + vec3( side * 0.5, 0.0, -h / 3.0);

    float tClosest = 1e30;
    vec3 bestNorm = vec3(0.0);
    bool found = false;

    vec3 pts[3];
    pts[0] = p0; pts[1] = p1; pts[2] = p2;

    for (int i = 0; i < 3; i++) {
        vec3 a = pts[i];
        vec3 b = pts[(i + 1) % 3];
        vec3 edge = b - a;
        vec3 sideNorm = normalize(vec3(edge.z, 0.0, -edge.x));

        float denom = dot(ray.direction, sideNorm);
        if (abs(denom) > 1e-6) {
            float t = dot(a - ray.origin, sideNorm) / denom;
            if (t > 0.001 && t < tClosest) {
                vec3 p = ray.origin + ray.direction * t;
                if (p.y >= baseCenter.y && p.y <= baseCenter.y + height) {
                    vec3 ap = p - a;
                    float edgeLen = length(edge);
                    float proj = dot(ap, edge) / (edgeLen * edgeLen);
                    if (proj >= 0.0 && proj <= 1.0) {
                        tClosest = t;
                        bestNorm = sideNorm;
                        found = true;
                    }
                }
            }
        }
    }

    if (abs(ray.direction.y) > 1e-6) {
        float tTop = (baseCenter.y + height - ray.origin.y) / ray.direction.y;
        if (tTop > 0.001 && tTop < tClosest) {
            vec3 p = ray.origin + ray.direction * tTop;
            vec2 v0 = p2.xz - p0.xz;
            vec2 v1 = p1.xz - p0.xz;
            vec2 v2 = p.xz - p0.xz;
            float dot00 = dot(v0, v0);
            float dot01 = dot(v0, v1);
            float dot02 = dot(v0, v2);
            float dot11 = dot(v1, v1);
            float dot12 = dot(v1, v2);
            float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
            if (u >= 0.0 && v >= 0.0 && (u + v) <= 1.0) {
                tClosest = tTop;
                bestNorm = vec3(0.0, 1.0, 0.0);
                found = true;
            }
        }
        float tBot = (baseCenter.y - ray.origin.y) / ray.direction.y;
        if (tBot > 0.001 && tBot < tClosest) {
            vec3 p = ray.origin + ray.direction * tBot;
            vec2 v0 = p2.xz - p0.xz;
            vec2 v1 = p1.xz - p0.xz;
            vec2 v2 = p.xz - p0.xz;
            float dot00 = dot(v0, v0);
            float dot01 = dot(v0, v1);
            float dot02 = dot(v0, v2);
            float dot11 = dot(v1, v1);
            float dot12 = dot(v1, v2);
            float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
            if (u >= 0.0 && v >= 0.0 && (u + v) <= 1.0) {
                tClosest = tBot;
                bestNorm = vec3(0.0, -1.0, 0.0);
                found = true;
            }
        }
    }

    if (found) {
        tHit = tClosest;
        hitNormal = bestNorm;
        return true;
    }
    tHit = 0.0;
    hitNormal = vec3(0.0);
    return false;
}

bool intersectTriangle(
    Ray ray,
    vec3 v0, vec3 v1, vec3 v2,
    vec3 n0, vec3 n1, vec3 n2,
    out float tHit, out vec3 hitNormal
) {
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 pvec = cross(ray.direction, e2);
    float det = dot(e1, pvec);

    if (abs(det) < 1e-8) {
        tHit = 0.0;
        hitNormal = vec3(0.0);
        return false;
    }
    float invDet = 1.0 / det;

    vec3 tvec = ray.origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    vec3 qvec = cross(tvec, e1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0 || (u + v) > 1.0) return false;

    float t = dot(e2, qvec) * invDet;
    if (t < 0.0005) return false;

    tHit = t;
    float w = 1.0 - u - v;
    vec3 N = normalize(w * n0 + u * n1 + v * n2);
    if (dot(N, ray.direction) > 0.0) N = -N;
    hitNormal = N;
    return true;
}

void getSurfaceGeometry(uint surfaceId, vec2 uv, out vec3 pos, out vec3 nor) {
    float u = clamp(uv.x, 0.001, 0.999);
    float v = clamp(uv.y, 0.001, 0.999);
    if (surfaceId == 0u) {
        pos = vec3(mix(kRoomMinX, kRoomMaxX, u), 0.002, mix(kRoomMinZ, kRoomMaxZ, v));
        nor = vec3(0.0, 1.0, 0.0);
    } else if (surfaceId == 1u) {
        pos = vec3(mix(kRoomMinX, kRoomMaxX, u), kRoomMaxY - 0.002, mix(kRoomMinZ, kRoomMaxZ, v));
        nor = vec3(0.0, -1.0, 0.0);
    } else if (surfaceId == 2u) {
        pos = vec3(mix(kRoomMinX, kRoomMaxX, u), mix(kRoomMinY + 0.002, kRoomMaxY - 0.002, v), kRoomMaxZ - 0.002);
        nor = vec3(0.0, 0.0, -1.0);
    } else if (surfaceId == 3u) {
        pos = vec3(kRoomMinX + 0.002, mix(kRoomMinY + 0.002, kRoomMaxY - 0.002, v), mix(kRoomMinZ, kRoomMaxZ, u));
        nor = vec3(1.0, 0.0, 0.0);
    } else {
        pos = vec3(kRoomMaxX - 0.002, mix(kRoomMinY + 0.002, kRoomMaxY - 0.002, v), mix(kRoomMinZ, kRoomMaxZ, u));
        nor = vec3(-1.0, 0.0, 0.0);
    }
}
