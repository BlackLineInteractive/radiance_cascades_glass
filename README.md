# Radiance Cascades Glass & Caustics

A high-performance, multi-backend real-time ray tracing engine demonstrating **Radiance Cascades for Dielectrics**, **Spectral Cauchy Dispersion**, and **Atomic GPU Caustics**. Built with native implementations in **Apple Metal**, **Vulkan 1.2+**, and **OpenGL 4.3+ Core**.

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

We express our sincere gratitude and respect to Alexander Sannikov and the real-time computer graphics community for publishing this breakthrough paradigm and demonstrating hierarchical angular-spatial radiance representations. This project builds upon these mathematical foundations to study how cascading radiance integrates with complex curved glass, spectral dispersion, and atomic photon splatting.

---

## Author & Connect

Created and maintained by **Blackline Interactive**:

- **Instagram**: [@blacklineinteractive](https://www.instagram.com/blacklineinteractive)
- **Telegram**: [t.me/blacklineinteractive](https://t.me/blacklineinteractive)
- **YouTube**: [@blacklineinteractive](https://youtube.com/@blacklineinteractive)
- **LinkedIn**: [linkedin.com/in/blacklineinteractive](http://linkedin.com/in/blacklineinteractive)

---

## Technical Overview

Traditional real-time ray tracing struggles with complex dielectric phenomena (such as multi-interface refraction, chromatic dispersion, rough transmission for frosted glass, and focused photon caustics) due to high sampling noise and prohibitive computational cost.

This engine unifies:

1. **3D Radiance Cascades Global Illumination**: Hierarchical 4-cascade angular-spatial radiance representation with bounded distance intervals and far-to-near merging across surfaces for smooth indirect bounce lighting and color bleeding. *(Note: RC is utilized here to calculate diffuse irradiance, which is then integrated with complex dielectric phenomena like specular refraction and rough transmission).*
2. **Multi-Wavelength Cauchy Dispersion**: Spectral splitting ($R, G, B$) through Newton's prism and crystal spheres using Cauchy's dispersion equation:
   $$n(\lambda) = n_0 + \frac{B}{\lambda^2}$$
3. **Atomic Caustic Splatting**: Parallel forward photon projection from directional sun rays, refracted through complex glass geometries and accumulated into a 32-bit fixed-point spatial irradiance grid using 32-bit GPU atomics (`atomicAdd`).
4. **Beer-Lambert Absorption**: Physically accurate volume attenuation along internal ray paths:
   $$I(d) = I_0 \exp(-\alpha d)$$
5. **GPU Linear BVH Traversal**: Fixed-depth (64-entry stack) linear bounding volume hierarchy traversal with near-child sorting heuristic intersecting the 6,320-triangle Utah Teapot alongside analytical geometric primitives (spheres, cylinders, prisms, slabs).

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

The engine is engineered as a clean, decoupled system with shared mathematical foundations and asset definitions:

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
| **Performance** | ~70 FPS (AMD 5500M) | ~48 FPS (MoltenVK) | Native on Linux/Win |

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

All backends support automated headless execution for profiling, CI/CD validation, and snapshot generation:

```bash
# Run 4-mode automated headless benchmark on Metal
./Metal/rc_glass_app --headless

# Run 4-mode automated headless benchmark on Vulkan
./Vulkan/rc_glass_vk --headless

# Run on OpenGL
./OpenGL/rc_glass_gl --headless
```

### CLI Arguments

- `--headless` or `--benchmark`: Runs all 4 optical modes sequentially, records microsecond-accurate frametimes, logs FPS, and outputs PNG renders into `output/`.
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

### 4. 3D Radiance Cascades Formulation

Following Alexander Sannikov's Radiance Cascades framework, indirect radiance is evaluated across a 4-level hierarchy ($C = 4$) with bounded geometric range intervals:

$$I_c = [r_c, r_{c+1}], \quad \mathbf{r} = \{0.005\,\text{m},\, 0.25\,\text{m},\, 0.80\,\text{m},\, 2.50\,\text{m},\, 100.0\,\text{m}\}$$

Each cascade balances spatial probe density and angular ray resolution ($M_c \in \{16, 32, 64, 128\}$):

- **Interval-Bounded Ray Tracing**: Rays for cascade $c$ are traced exclusively within distance interval $[r_c, r_{c+1}]$, drastically pruning BVH traversal and primitive intersections.
- **Hierarchical Far-to-Near Merging**: Radiance is merged backwards from Cascade 3 down to Cascade 0:
  $$L_c(\vec{\omega}) = L_c^{\text{local}}(\vec{\omega}) + \tau_c(\vec{\omega}) \cdot L_{c+1}^{\text{merged}}(\vec{\omega})$$
  where $\tau_c(\vec{\omega}) = 0.0$ if occluded by geometry in interval $[r_c, r_{c+1}]$, and $\tau_c(\vec{\omega}) = 1.0$ (or boundary-faded) if unoccluded.
- **Cosine-Weighted Irradiance Integration**: The fully merged Cascade 0 radiance field integrates into surface irradiance:
  $$E(\mathbf{x}) = \frac{1}{\pi} \sum_{k=0}^{M_0 - 1} L_0(\vec{\omega}_k) \cos\theta_k \Delta\Omega_k$$

---

## License

This project is open-source software licensed under the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Blackline Interactive.
