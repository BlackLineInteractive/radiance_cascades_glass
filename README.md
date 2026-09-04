# Radiance Cascades Glass & Caustics

A real-time ray tracer for glass: refraction with dispersion, forward-splatted floor caustics, and a cascaded irradiance cache for the indirect bounce. Three separate backends — Apple Metal, Vulkan 1.2+, and OpenGL 4.3+ Core — share the scene definition and the mesh loader.

![Realistic Glass Cascade GI](media/1_Realistic_Glass_Cascade_GI.png)

---

## Important Notice & Disclaimer

> [!WARNING]
> **Experimental Demo / Proof of Concept**
> This repository is an **experimental research demo and educational prototype**. It explores the application of Radiance Cascades principles to complex dielectric transmission and real-time caustic generation. Because it is a proof-of-concept, it may contain physical simplifications, mathematical approximations, edge-case inaccuracies, or implementation bugs. It is not intended as a drop-in production rendering library.

> [!NOTE]
> **Developer Note & AI Assistance**
> AI was used to assist in writing this code due to the high barrier of entry and complexity of Vulkan, as well as to help grasp the intricate mathematics behind Radiance Cascades. It is disheartening to see a double standard in the graphics community where massive corporations are praised for AI generation (e.g., DLSS 5), while solo developers releasing completely free, open-source code with full author attribution are heavily criticized. I am learning, sharing my journey, and hoping to make these complex techniques more accessible.

---

## Attribution & Credits

The **Radiance Cascades** algorithm was conceived and pioneered by **Alexander Sannikov**, who introduced the concept in his 2023/2024 research:

- **Alexander Sannikov** - *"Radiance Cascades: A Novel Approach to Calculating Global Illumination"* (2023/2024).
- Repository: [https://github.com/Raikiri/RadianceCascadesPaper](https://github.com/Raikiri/RadianceCascadesPaper)
- Direct PDF: [RadianceCascades.pdf](https://github.com/Raikiri/RadianceCascadesPaper/blob/main/out_latexmk2/RadianceCascades.pdf)

What this project borrows from that work is the interval partition and the far-to-near merge. It is not a faithful implementation — see [What is and isn't Radiance Cascades here](#what-is-and-isnt-radiance-cascades-here) for where it departs.

---

## Author & Connect

Created and maintained by **Blackline Interactive**:

- **Instagram**: [@blacklineinteractive](https://www.instagram.com/blacklineinteractive)
- **Telegram**: [t.me/blacklineinteractive](https://t.me/blacklineinteractive)
- **YouTube**: [@blacklineinteractive](https://youtube.com/@blacklineinteractive)
- **LinkedIn**: [linkedin.com/in/blacklineinteractive](http://linkedin.com/in/blacklineinteractive)

---

## Technical Overview

Glass is awkward for a real-time path tracer: refraction through two interfaces is specular, so the paths that matter are exactly the ones importance sampling finds slowly, and the caustics they produce are the noisiest part of the image. This demo sidesteps that by never sampling those paths backwards.

The frame is five compute passes:

1. **Cascaded irradiance** — a 64x64 probe grid per room surface (five surfaces packed into one 320x64 atlas) gathers indirect light over four non-overlapping distance intervals, merged far-to-near. Result is diffuse irradiance only; specular refraction is handled separately in the shading pass.
2. **Atlas filter** — 7x7 Gaussian over each surface, so probe noise does not show up as blotches on the walls.
3. **Caustic splatting** — one photon per thread, refracted through a glass object and projected onto the floor. Because photons land wherever they land, the accumulation buffer is integer and the splat is a bilinear `atomic_fetch_add` into fixed point.
4. **Caustic filter** — reads the integer buffer back into a float texture, with an extra roughness-driven blur in frosted mode.
5. **Shading** — one primary ray per pixel. Glass gets a Fresnel-weighted split between one reflection ray and one refraction ray traced per channel (R/G/B use different IOR, which is what produces the coloured fringes).

Dispersion uses a Cauchy-style split, $n(\lambda) = n_0 + B/\lambda^2$, collapsed to three fixed offsets rather than a real spectral sampling. Internal attenuation is Beer-Lambert, $I(d) = I_0 e^{-\alpha d}$. The teapot (6,320 triangles) sits in a 4,095-node linear BVH traversed with a 64-entry stack and near-child ordering; everything else in the scene is an analytic primitive.

---

## Visual Gallery

| Mode 1: Clear Glass & Floor Caustics | Mode 2: Frosted / Rough Glass |
| :---: | :---: |
| ![Realistic](media/1_Realistic_Glass_Cascade_GI.png) | ![Frosted](media/2_Frosted_Glass.png) |
| *Snell refraction, Cauchy dispersion, floor photon caustics* | *Cone-jittered transmission & diffused caustic filter* |

| Mode 3: Newton Prism Dispersion | Mode 0: Whitted RT Baseline |
| :---: | :---: |
| ![Dispersion](media/3_Spectral_Dispersion.png) | ![Whitted Baseline](media/4_Whitted_Baseline.png) |
| *Amplified spectral separation on prism & crystal sphere* | *Classical binary shadow ray (zero caustics, dark shadow)* |

---

## Optical Modes

| Mode | Identifier | Description |
| :---: | :--- | :--- |
| **Mode 1** | **Realistic Glass + Caustics** | Snell refraction, Fresnel reflection, Cauchy spectral dispersion, floor caustics splatting, and radiance cascade GI. |
| **Mode 2** | **Frosted / Rough Glass** | Micro-roughness transmission cone sampling, softened refraction, and Gaussian-diffused caustic footprints. |
| **Mode 3** | **High Spectral Dispersion** | Exaggerated Cauchy coefficients on crystal spheres and Newton's triangular prism, displaying distinct spectral separation. |
| **Mode 0** | **Whitted Baseline** | Classical recursive ray tracing with binary shadow testing (no caustics, dark glass shadows). |

---

## Architecture & Backends

The three backends are independent hosts over a shared scene definition and mesh loader:

```
radiance_cascades_glass/
├── assets/
│   ├── teapot.bin              # 6,320 triangles, 4,095 BVH nodes binary mesh
│   └── TeapotData.h            # Binary-aligned C++17 loader & GPU structs (48B BVH, 96B tri)
├── common/
│   ├── Camera.h                # Unified OrbitCamera (Apple SIMD & GLM)
│   ├── scene_rt.glsl           # Shared GLSL constants, structs & tone mapping
│   └── scene_intersect.glsl    # BVH traversal & analytical intersection routines
├── Metal/
│   ├── src/main.mm             # Native Cocoa / MetalKit application host
│   ├── shaders/                # Metal Shading Language compute kernels
│   └── build.sh                # Standalone Metal build script
├── Vulkan/
│   ├── src/main.cpp            # Vulkan 1.2+ compute pipeline host (GLFW)
│   ├── shaders/                # GLSL compute shaders (compiled to SPIR-V)
│   ├── CMakeLists.txt          # Standalone Vulkan CMake configuration
│   └── build.sh                # Standalone Vulkan build script
├── OpenGL/
│   ├── src/main.cpp            # OpenGL 4.3+ Core host with compute dispatches
│   ├── src/gl_loader.h/.cpp    # Standalone GL 4.3 function loader (GLFW_INCLUDE_NONE)
│   ├── shaders/                # OpenGL 4.3 Core GLSL compute shaders
│   ├── CMakeLists.txt          # Standalone OpenGL CMake configuration
│   └── build.sh                # Standalone OpenGL build script
├── media/                      # High-resolution benchmark renders
├── output/                     # Benchmark artifacts and snapshot exports
└── CMakeLists.txt              # Unified root build configuration
```

### Backend Comparison

| Feature | Apple Metal | Vulkan 1.2+ | OpenGL 4.3+ Core |
| :--- | :---: | :---: | :---: |
| **Language** | MSL (C++14 based) | GLSL $\to$ SPIR-V | GLSL 430 / 450 |
| **Compute Passes** | 5 Pipelines | 5 Pipelines | 5 Programs |
| **Memory Barriers** | Implicit / Metal Fences | Explicit `VkMemoryBarrier` | `glMemoryBarrier` |
| **Shader Storage** | `device const T*` | SSBO (`std430`) | SSBO (`std430`) |
| **Platform Target** | macOS (Native) | Cross-platform / MoltenVK | Linux / Windows / Mesa |
| **Measured (1080p)** | 24 fps clear / 20 fps frosted, AMD Radeon Pro 5500M | not benchmarked | not benchmarked |

---

## Building and Running

### Prerequisites

#### macOS

```bash
# Install Homebrew dependencies
brew install cmake glfw glm vulkan-headers vulkan-loader molten-vk glslang
```

#### Ubuntu / Debian Linux

```bash
sudo apt update
sudo apt install -y cmake g++ libvulkan-dev vulkan-tools libglfw3-dev libglm-dev glslang-tools
```

#### Windows

Install the [Vulkan SDK](https://vulkan.lunarg.com/), [CMake](https://cmake.org/), and install `glfw` and `glm` via [vcpkg](https://github.com/microsoft/vcpkg).

---

### Method 1: Unified CMake (Recommended)

From the project root:

```bash
cmake -B build -S .
cmake --build build -j
```

This automatically detects available SDKs and compiles:

- `build/rc_glass_app` (Metal on macOS)
- `build/Vulkan/rc_glass_vk` (Vulkan)
- `build/OpenGL/rc_glass_gl` (OpenGL 4.3+)

---

### Method 2: Standalone Backend Scripts

Each backend includes an independent build script:

```bash
# Build & run Apple Metal
cd Metal && ./build.sh && ./rc_glass_app

# Build & run Vulkan
cd Vulkan && ./build.sh && ./rc_glass_vk

# Build & run OpenGL (Linux / Windows / Mesa)
cd OpenGL && ./build.sh && ./rc_glass_gl
```

---

## Interactive Controls

| Control | Action |
| :--- | :--- |
| **Left Click + Drag** | Orbit camera around target |
| **Option + Drag** | Pan camera horizontally & vertically |
| **Right Click + Drag / Scroll** | Zoom camera in / out |
| **Space** | Toggle animated sun orbit |
| **Arrow Keys** | Manually adjust sun elevation and azimuth |
| **1** | Switch to **Mode 1** (Realistic Clear Glass + Caustics) |
| **2** | Switch to **Mode 2** (Frosted Rough Glass) |
| **3** | Switch to **Mode 3** (High Spectral Dispersion Prism) |
| **0** | Switch to **Mode 0** (Whitted Ray Tracing Baseline) |
| **+ / -** | Increase / decrease glass surface roughness |
| **R** | Reset camera to default perspective |
| **S** | Capture high-resolution screenshot to `output/` |
| **Esc** | Exit application |

---

## Headless Benchmarking & CLI Options

Every backend takes `--headless`, which renders all four modes at 1920x1080 and writes PNGs to `output/`:

```bash
# Run 4-mode automated headless benchmark on Metal
./Metal/rc_glass_app --headless

# Run 4-mode automated headless benchmark on Vulkan
./Vulkan/rc_glass_vk --headless

# Run on OpenGL
./OpenGL/rc_glass_gl --headless
```

### CLI Arguments

- `--headless` or `--benchmark`: Renders all four modes, timing 20 frames each after 3 warm-up frames, and writes PNGs to `output/`.
- `--teapot <path>`: Specifies custom path to `teapot.bin` mesh data.
- `--shader <path>`: (Metal only) Specifies custom compiled `.metallib` path.

---

## Mathematical Formulations

### 1. Dielectric Fresnel Equations

For unpolarized light with incident angle $\theta_i$ and transmitted angle $\theta_t$:
$$R_s = \left|\frac{n_1 \cos\theta_i - n_2 \cos\theta_t}{n_1 \cos\theta_i + n_2 \cos\theta_t}\right|^2, \quad R_p = \left|\frac{n_1 \cos\theta_t - n_2 \cos\theta_i}{n_1 \cos\theta_t + n_2 \cos\theta_i}\right|^2$$
$$F(\theta_i) = \frac{1}{2} (R_s + R_p)$$

### 2. Snell-Descartes Refraction Vector

Given incident unit direction $\mathbf{I}$, surface normal $\mathbf{N}$, and relative index $\eta = n_1 / n_2$:
$$\cos\theta_i = -\mathbf{N} \cdot \mathbf{I}, \quad \sin^2\theta_t = \eta^2 (1 - \cos^2\theta_i)$$
$$\mathbf{T} = \eta \mathbf{I} + (\eta \cos\theta_i - \sqrt{1 - \sin^2\theta_t}) \mathbf{N}$$
If $\sin^2\theta_t > 1$, Total Internal Reflection (TIR) occurs.

### 3. Bilinear Photon Splatting

Photons intersecting the receiver plane $(x, z)$ distribute energy to surrounding grid cells $(x_0, z_0), (x_1, z_0), (x_0, z_1), (x_1, z_1)$ with bilinear weights:
$$w_{00} = (1 - f_x)(1 - f_z), \quad w_{10} = f_x (1 - f_z), \quad w_{01} = (1 - f_x) f_z, \quad w_{11} = f_x f_z$$
Accumulated into integer SSBO buffers via fixed-point scaling factor $S = 10^9$:
$$\Delta I = \lfloor \Phi \cdot w_{uv} \cdot S \rfloor$$

### 4. Cascaded Irradiance

Four cascades, each owning one segment of the ray:

$$I_c = [r_c, r_{c+1}], \quad \mathbf{r} = \{0.005,\ 0.25,\ 0.80,\ 2.50,\ 100.0\}\ \text{m}$$

with $M_c \in \{16, 32, 64, 128\}$ directions and probes snapped to a $2^c$-texel grid on the atlas. A cascade-$c$ ray is traced only inside its own interval, so nothing is intersected twice, and the results merge back down:

$$L_c(\vec{\omega}_i) = L_c^{\text{local}}(\vec{\omega}_i) + \tau_c(\vec{\omega}_i) \cdot \tfrac{1}{2}\left(L_{c+1}(\vec{\omega}_{2i}) + L_{c+1}(\vec{\omega}_{2i+1})\right)$$

$\tau_c$ is the residual transmittance: 1 when the ray leaves the interval unobstructed, and `smoothstep(0.85, 1, t)` when it hits near the far edge, which stops the interval boundary from showing up as a hard ring. Cascade 3's residual picks up the sky instead of a further cascade.

Irradiance is the mean of the merged cascade-0 radiance:

$$E(\mathbf{x}) = \frac{1}{M_0}\sum_{k=0}^{M_0-1} L_0(\vec{\omega}_k)$$

There is no explicit $\cos\theta$ term because the directions are drawn cosine-weighted, so the estimator is already the cosine-weighted average.

---

## What is and isn't Radiance Cascades here

The interval partition and the far-to-near merge come straight from Sannikov's formulation. Three things do not, and calling the result "Radiance Cascades" without qualification would be overselling it:

**Cascades are not stored, so nothing is amortised.** In the real scheme, a coarse cascade is computed once at low spatial resolution and then read by every fine probe under it — that is where the speedup comes from. Here the atlas is one thread per texel and each thread walks all four cascades itself. Texels in the same 8x8 block do share cascade 3's *probe position*, but they each re-trace its 128 rays with their own jitter, so the coarse level costs 64x what the structure is supposed to make it cost. In practice it behaves as supersampling, not as a cache.

**The angular ratio is off by 2x per level.** Spatial resolution drops 4x in area per cascade while the ray count only doubles. Sannikov's penumbra condition wants angular resolution to grow as fast as spatial resolution shrinks; at 2x, the higher cascades are angularly under-resolved for the solid angle they cover.

**No bilinear merge between coarse probes.** Each texel reads a single snapped coarse probe rather than interpolating the four nearest, which is visible as blocking at cascade boundaries on large flat surfaces.

There is also a structural limit: probes only exist on the five room surfaces, as a 64x64 lightmap each. So this is a surface irradiance cache with a cascaded gather, not a volumetric or screen-space cascade hierarchy. Glass objects have no probes of their own and read a normal-weighted blend of the five walls.

---

## License

This project is open-source software licensed under the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Blackline Interactive.
