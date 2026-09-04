bool intersectTeapotBVH(Ray ray, uint numNodes, out float tHit, out vec3 hitNormal) {
    if (numNodes == 0u) return false;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    float tClosest = 1e30;
    vec3 bestNormal = vec3(0.0);
    bool hitAny = false;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        GPUBVHNode node = bvhNodes[nodeIdx];

        float tBox;
        if (!intersectBoxFast(ray, node.bmin.xyz, node.bmax.xyz, tBox)) continue;
        if (tBox >= tClosest) continue;

        if (node.leftChild < 0) {
            int triCount = -node.leftChild;
            int triStart = node.rightChild;
            for (int i = 0; i < triCount; i++) {
                GPUTriangle tri = triangles[triStart + i];
                float tTri;
                vec3 nTri;
                if (intersectTriangle(ray, tri.v0.xyz, tri.v1.xyz, tri.v2.xyz,
                                      tri.n0.xyz, tri.n1.xyz, tri.n2.xyz, tTri, nTri)) {
                    if (tTri < tClosest) {
                        tClosest = tTri;
                        bestNormal = nTri;
                        hitAny = true;
                    }
                }
            }
        } else {
            float tNearL, tNearR;
            bool hitL = intersectBoxFast(ray, bvhNodes[node.leftChild].bmin.xyz, bvhNodes[node.leftChild].bmax.xyz, tNearL);
            bool hitR = intersectBoxFast(ray, bvhNodes[node.rightChild].bmin.xyz, bvhNodes[node.rightChild].bmax.xyz, tNearR);

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

HitRecord intersectScene(Ray ray, bool testGlass, uint numNodes) {
    HitRecord hit;
    hit.hit = false;
    hit.distance = 1e30;
    hit.isGlass = false;
    hit.roughness = 0.0;
    hit.objectId = 0u;

    float t;
    vec3 norm;

    if (abs(ray.direction.z) > 1e-5) {
        t = (kRoomMaxZ - ray.origin.z) / ray.direction.z;
        if (t > 0.001 && t < hit.distance) {
            vec3 p = ray.origin + ray.direction * t;
            if (p.x >= kRoomMinX && p.x <= kRoomMaxX && p.y >= kRoomMinY && p.y <= kRoomMaxY) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = vec3(0.0, 0.0, -1.0);
                hit.albedo = vec3(0.88, 0.86, 0.82);
                hit.roughness = 0.9;
                hit.isGlass = false;
                hit.objectId = 1u;
            }
        }
    }

    if (abs(ray.direction.y) > 1e-5) {
        t = -ray.origin.y / ray.direction.y;
        if (t > 0.001 && t < hit.distance) {
            vec3 p = ray.origin + ray.direction * t;
            if (p.x >= kRoomMinX && p.x <= kRoomMaxX && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = vec3(0.0, 1.0, 0.0);
                float tileX = fract(p.x * 1.5);
                float tileZ = fract(p.z * 1.5);
                float grout = (tileX < 0.03 || tileZ < 0.03) ? 0.45 : 1.0;
                hit.albedo = vec3(0.72, 0.70, 0.65) * grout;
                hit.roughness = 0.4;
                hit.isGlass = false;
                hit.objectId = 2u;
            }
        }

        t = (kRoomMaxY - ray.origin.y) / ray.direction.y;
        if (t > 0.001 && t < hit.distance) {
            vec3 p = ray.origin + ray.direction * t;
            if (p.x >= kRoomMinX && p.x <= kRoomMaxX && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = vec3(0.0, -1.0, 0.0);
                hit.albedo = vec3(0.92, 0.92, 0.90);
                hit.roughness = 0.9;
                hit.isGlass = false;
                hit.objectId = 3u;
            }
        }
    }

    if (abs(ray.direction.x) > 1e-5) {
        t = (kRoomMinX - ray.origin.x) / ray.direction.x;
        if (t > 0.001 && t < hit.distance) {
            vec3 p = ray.origin + ray.direction * t;
            if (p.y >= kRoomMinY && p.y <= kRoomMaxY && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                if (p.y >= kWinMinY && p.y <= kWinMaxY && p.z >= kWinMinZ && p.z <= kWinMaxZ) {
                    bool isMuntin = (abs(p.y - 1.80) < 0.045) ||
                                    (abs(p.z - 0.00) < 0.045) ||
                                    (abs(p.z - 1.05) < 0.040) ||
                                    (abs(p.z + 1.05) < 0.040) ||
                                    (abs(p.y - kWinMinY) < 0.06) ||
                                    (abs(p.y - kWinMaxY) < 0.06) ||
                                    (abs(p.z - kWinMinZ) < 0.06) ||
                                    (abs(p.z - kWinMaxZ) < 0.06);
                    if (isMuntin) {
                        hit.hit = true;
                        hit.distance = t;
                        hit.position = p;
                        hit.normal = vec3(1.0, 0.0, 0.0);
                        hit.albedo = vec3(0.24, 0.16, 0.10);
                        hit.roughness = 0.6;
                        hit.isGlass = false;
                        hit.objectId = 6u;
                    }
                } else {
                    hit.hit = true;
                    hit.distance = t;
                    hit.position = p;
                    hit.normal = vec3(1.0, 0.0, 0.0);
                    hit.albedo = vec3(0.85, 0.22, 0.20);
                    hit.roughness = 0.85;
                    hit.isGlass = false;
                    hit.objectId = 4u;
                }
            }
        }

        t = (kRoomMaxX - ray.origin.x) / ray.direction.x;
        if (t > 0.001 && t < hit.distance) {
            vec3 p = ray.origin + ray.direction * t;
            if (p.y >= kRoomMinY && p.y <= kRoomMaxY && p.z >= kRoomMinZ && p.z <= kRoomMaxZ) {
                hit.hit = true;
                hit.distance = t;
                hit.position = p;
                hit.normal = vec3(-1.0, 0.0, 0.0);
                hit.albedo = vec3(0.18, 0.75, 0.40);
                hit.roughness = 0.85;
                hit.isGlass = false;
                hit.objectId = 5u;
            }
        }
    }

    if (testGlass) {
        if (intersectTeapotBVH(ray, numNodes, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = vec3(1.0);
                hit.roughness = 0.0;
                hit.isGlass = true;
                hit.objectId = 10u;
                hit.ior = 1.52;
                hit.dispersion = 0.025;
                hit.absorption = vec3(0.04, 0.04, 0.04);
            }
        }

        if (intersectSphere(ray, kSphereCenter, kSphereRadius, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = vec3(1.0);
                hit.roughness = 0.0;
                hit.isGlass = true;
                hit.objectId = 11u;
                hit.ior = 1.62;
                hit.dispersion = 0.040;
                hit.absorption = vec3(0.02, 0.02, 0.02);
            }
        }

        if (intersectCylinder(ray, kCylinderCenter, kCylinderRadius, kCylinderHeight, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = vec3(1.0);
                hit.roughness = 0.0;
                hit.isGlass = true;
                hit.objectId = 12u;
                hit.ior = 1.50;
                hit.dispersion = 0.010;
                hit.absorption = vec3(1.20, 0.15, 0.90) * 2.2;
            }
        }

        if (intersectTriangularPrism(ray, kPrismCenter, kPrismSide, kPrismHeight, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = vec3(1.0);
                hit.roughness = 0.0;
                hit.isGlass = true;
                hit.objectId = 13u;
                hit.ior = 1.58;
                hit.dispersion = 0.055;
                hit.absorption = vec3(0.03, 0.03, 0.03);
            }
        }

        vec3 slabMin = vec3( 1.00, 0.0, 0.35);
        vec3 slabMax = vec3( 1.70, 0.06, 1.05);
        if (intersectBox(ray, slabMin, slabMax, t, norm)) {
            if (t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + ray.direction * t;
                hit.normal = norm;
                hit.albedo = vec3(0.92, 0.94, 0.96);
                hit.roughness = 0.35;
                hit.isGlass = true;
                hit.objectId = 14u;
                hit.ior = 1.52;
                hit.dispersion = 0.015;
                hit.absorption = vec3(0.2, 0.15, 0.1);
            }
        }
    }

    return hit;
}
