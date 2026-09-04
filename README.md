# Radiance Cascades Glass & Caustics

A real-time ray tracer for glass: refraction with dispersion, forward-splatted floor caustics, and a cascaded irradiance cache for the indirect bounce. Three separate backends - Apple Metal, Vulkan 1.2+, and OpenGL 4.3+ Core - share the scene definition and the mesh loader.

<p align="left">
  <a href="https://www.instagram.com/blacklineinteractive"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram" /></a>
  <a href="https://t.me/blacklineinteractive"><img src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram" /></a>
  <a href="https://youtube.com/@blacklineinteractive"><img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube" /></a>
  <a href="https://www.linkedin.com/in/blacklineinteractive"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
</p>

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

What this project borrows from that work is the interval partition and the far-to-near merge. It is not a faithful implementation - see [What is and isn't Radiance Cascades here](#what-is-and-isnt-radiance-cascades-here) for where it departs.

---

## Author & Connect

Created and maintained by **Blackline Interactive**:

<p align="left">
  <a href="https://www.instagram.com/blacklineinteractive"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram" /></a>
  <a href="https://t.me/blacklineinteractive"><img src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram" /></a>
  <a href="https://youtube.com/@blacklineinteractive"><img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube" /></a>
  <a href="https://www.linkedin.com/in/blacklineinteractive"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
</p>

| Platform | Link | Description |
| :--- | :--- | :--- |
| **Instagram** | [@blacklineinteractive](https://www.instagram.com/blacklineinteractive) | Visual dev logs & graphics demos |
| **Telegram** | [t.me/blacklineinteractive](https://t.me/blacklineinteractive) | Community & project updates |
| **YouTube** | [@blacklineinteractive](https://youtube.com/@blacklineinteractive) | Real-time benchmarks & video breakdowns |
| **LinkedIn** | [blacklineinteractive](https://www.linkedin.com/in/blacklineinteractive) | Professional network & engineering |

---

## Technical Overview

Glass is awkward for a real-time path tracer: refraction through two interfaces is specular, so the paths that matter are exactly the ones importance sampling finds slowly, and the caustics they produce are the noisiest part of the image. This demo sidesteps that by never sampling those paths backwards.

The frame is five compute passes:

1. **Cascaded irradiance** - a 64x64 probe grid per room surface (five surfaces packed into one 320x64 atlas) gathers indirect light over four non-overlapping distance intervals, merged far-to-near. Result is diffuse irradiance only; specular refraction is handled separately in the shading pass.
2. **Atlas filter** - 7x7 Gaussian over each surface, so probe noise does not show up as blotches on the walls.
3. **Caustic splatting** - one photon per thread, refracted through a glass object and projected onto the floor. Because photons land wherever they land, the accumulation buffer is integer and the splat is a bilinear `atomic_fetch_add` into fixed point.
4. **Caustic filter** - reads the integer buffer back into a float texture, with an extra roughness-driven blur in frosted mode.
5. **Shading** - one primary ray per pixel. Glass gets a Fresnel-weighted split between one reflection ray and one refraction ray traced per channel (R/G/B use different IOR, which is what produces the coloured fringes).

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

The three backends are independent hosts over a shared scene definition and mesh loader. They are not quite feature-identical: the Metal cascade gather traces glass and feeds the previous frame's atlas back in for a second bounce, while the GLSL one skips both and uses a flat ambient term.



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
| **Compute Passes** | 9 Pipelines | 9 Pipelines | 9 Programs |
| **Memory Barriers** | Implicit / Metal Fences | Explicit `VkMemoryBarrier` | `glMemoryBarrier` |
| **Shader Storage** | `device const T*` | SSBO (`std430`) | SSBO (`std430`) |
| **Platform Target** | macOS (Native) | Cross-platform / MoltenVK | Linux / Windows / Mesa |
| **Measured** | 69 fps clear / 47 fps frosted @1080p | 69 fps clear / 47 fps frosted @720p, via MoltenVK | not benchmarked (macOS caps GL at 4.1, no compute) |

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

Every backend takes `--headless`, which renders all four modes and writes PNGs to `output/` (Metal renders at 1080p, Vulkan and OpenGL at 720p):

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

Four cascades, each owning one segment of the ray and its own probe grid, stored in its own texture:

$$I_c = [r_c, r_{c+1}], \quad \mathbf{r} = \{0.005,\ 0.25,\ 0.80,\ 2.50,\ 100.0\}\ \text{m}$$

Probe density drops 4x in area per level up ($64^2, 32^2, 16^2, 8^2$ per surface), and ray count $M_c$ grows 4x to match ($M_c = 16 \cdot 4^c$, i.e. $16, 64, 256, 1024$), so every level spends the same total ray budget: $\text{probes}^2 \cdot M_c$ is constant across $c$. Each level is a separate compute dispatch, evaluated once, far-to-near - level 3 first (nothing above it, so its residual picks up the sky), then 2, 1, 0, each reading the level above rather than retracing it. A cascade-$c$ ray is traced only inside its own interval, so nothing is intersected twice, and the levels merge:

$$L_c(\vec{\omega}_i) = L_c^{\text{local}}(\vec{\omega}_i) + \tau_c(\vec{\omega}_i) \cdot \tfrac{1}{4}\sum_{k=0}^{3} \hat{L}_{c+1}(\vec{\omega}_{4i+k})$$

$\tau_c$ is the residual transmittance: 1 when the ray leaves the interval unobstructed, and `smoothstep(0.85, 1, t)` when it hits near the far edge, which stops the interval boundary from showing up as a hard ring. $\hat{L}_{c+1}$ is bilinearly interpolated across level $c+1$'s probe grid at the querying probe's position, not read from the single nearest coarse probe - a fine probe generally doesn't sit on a coarse one.

Irradiance is the mean of cascade 0's merged radiance:

$$E(\mathbf{x}) = \frac{1}{M_0}\sum_{k=0}^{M_0-1} L_0(\vec{\omega}_k)$$

There is no explicit $\cos\theta$ term because the directions are drawn cosine-weighted, so the estimator is already the cosine-weighted average.

---

## What is and isn't Radiance Cascades here

The interval partition and the far-to-near merge come straight from Sannikov's formulation. So does the amortization now: each cascade level is its own dispatch, over its own probe grid, evaluated once and read (not retraced) by the level below.

**Cascades are stored, one texture per level.** Level 3 (8x8 probes/surface, 1024 rays) is computed once and read by level 2's 16x16 probes, which is read by level 1's 32x32, down to level 0's 64x64. Ray count quadruples per level (16, 64, 256, 1024) to match the 4x drop in probe density, so every level spends the same total ray budget - `probes^2 * rays` is constant across levels, which is the condition the doubling in an earlier version of this engine was missing (it kept spatial resolution's 4x-per-level drop but only doubled the rays, leaving the coarse cascades angularly under-resolved for the solid angle they cover).

**Coarse probes are read with manual bilinear interpolation**, not snapped to the nearest one, since a fine probe rarely sits exactly on a coarse probe's position. GLSL can't express this as hardware texture filtering here because probes and directions share one image axis (bilinear filtering would blend across unrelated directions), so it's four explicit `imageLoad`s averaged by hand.

What this buys in practice: on the reference scene the old per-texel-retrace scheme spent about 4.9M ray-scene intersections a frame; the amortized version spends about 1.3M for the same four intervals at the corrected (4x) angular scaling - roughly 3.75x fewer traces while fixing the angular deficiency the old version had. On this machine (AMD Radeon Pro 5500M) the Metal cascade pass went from ~42ms to ~14.5ms.

There is still a structural limit that this pass doesn't touch: probes only exist on the five room surfaces, as a 64x64 lightmap each. So this is a surface irradiance cache with a cascaded gather, not a volumetric or screen-space cascade hierarchy. Glass objects have no probes of their own and read a normal-weighted blend of the five walls.

---

## License

This project is open-source software licensed under the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Blackline Interactive.
