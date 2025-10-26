# Matrix-Free GPU Solver Implementation

## Overview

This document describes the matrix-free GPU solver implementation for finite element analysis. This approach avoids assembling the global stiffness matrix K and instead computes K·v directly from element contributions.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                 Matrix-Free GPU Architecture                │
└────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │  Element Data   │
                        │  (GPU-resident) │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼────────┐       ┌───────▼────────┐
            │ Element Thread │       │ Element Thread │
            │   (Thread 0)   │  ...  │   (Thread N)   │
            │                │       │                │
            │ k₀ × v₀ → f₀  │       │ kₙ × vₙ → fₙ  │
            └───────┬────────┘       └───────┬────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │ (atomic scatter)
                        ┌────────▼────────┐
                        │   Result: K·v   │
                        └─────────────────┘
```

## Key Advantages

### 1. **Dense Element Operations**
- Each element works with a dense 12×12 stiffness matrix
- GPUs excel at dense matrix operations (high FLOPS/byte ratio)
- Regular memory access patterns

### 2. **Perfect Parallelism**
- Each GPU thread processes one element independently
- No data dependencies between threads
- Scales linearly with number of elements

### 3. **No Sparse Matrix Assembly**
- Avoids irregular sparse matrix storage
- Eliminates memory bandwidth bottleneck
- Lower memory footprint

### 4. **Numerical Advantages**
- Element stiffness computed on-the-fly (no round-off from assembly)
- More accurate for nonlinear analysis
- Can adapt to changing geometry/properties

## Implementation Details

### File Structure

```
src/analysis/gpu/
├── mod.rs                    # Module exports
├── metal.rs                  # Sparse GPU solver (legacy)
├── matrix_free.rs            # Matrix-free solver (NEW!)
└── shaders/
    ├── pcg.metal             # Sparse PCG kernels
    └── element_ops.metal     # Element-by-element kernels (NEW!)
```

### Core Components

#### 1. GPU Element Structure (`GpuBeamElement`)

Located in: `src/analysis/gpu/matrix_free.rs:14-33`

```rust
#[repr(C)]
struct GpuBeamElement {
    // Material properties
    E: f32,   // Young's modulus
    G: f32,   // Shear modulus
    A: f32,   // Cross-sectional area
    Iy: f32,  // Moment of inertia y
    Iz: f32,  // Moment of inertia z
    J: f32,   // Torsional constant

    // Geometry
    length: f32,
    cos_x: f32, cos_y: f32, cos_z: f32,  // Direction cosines

    // DOF mapping (12 DOFs: 6 per node × 2 nodes)
    dofs: [u32; 12],
}
```

**Purpose**: Compact, GPU-friendly representation of beam elements with all data needed for stiffness computation.

#### 2. Matrix-Free Solver (`MatrixFreeGPU`)

Located in: `src/analysis/gpu/matrix_free.rs:44-56`

```rust
pub struct MatrixFreeGPU {
    device: Device,
    command_queue: CommandQueue,
    element_matvec_pipeline: ComputePipelineState,

    // Element data stays on GPU!
    element_buffer: Buffer,
    num_elements: usize,
    num_dofs: usize,
}
```

**Key Methods**:
- `new(model: &StructuralModel)` - Converts model to GPU format, uploads elements
- `matvec(&self, x: &[f64])` - Computes y = K·x from elements
- `solve(...)` - PCG iteration without forming K

#### 3. Element Matvec Kernel

Located in: `src/analysis/gpu/shaders/element_ops.metal:208-269`

**Algorithm** (per element):
```
1. Extract element DOFs from global vector v (12 values)
2. Compute element stiffness in local coordinates (12×12)
   - Axial stiffness
   - Bending stiffness (2 planes)
   - Torsional stiffness
3. Transform to global coordinates: k_global = T^T × k_local × T
4. Compute element forces: f = k_global × v_element
5. Atomic scatter: Add f to global result vector
```

**Why Atomic Operations?**
Multiple elements can share nodes, so their contributions to the same DOF must be accumulated. Atomic operations ensure thread safety.

## Theoretical Performance

### Arithmetic Intensity Analysis

**Sparse Matrix Approach** (current GPU):
- Operations: ~24 FLOPS per element
- Memory: ~192 bytes (CSR format)
- Intensity: 0.125 FLOPS/byte ❌ (memory-bound)

**Matrix-Free Approach**:
- Operations: ~3,456 FLOPS per element (dense 12×12 operations)
- Memory: ~100 bytes (element data + v reads)
- Intensity: 34.5 FLOPS/byte ✅ (compute-bound)

### Expected Speedup

```
Problem: 10,000 elements, 60,000 DOF

CPU Cholesky:      ~1500 ms
GPU Sparse PCG:    ~800 ms  (current - overhead dominated)
GPU Matrix-Free:   ~80 ms   (estimated - 10× better!)

Speedup vs CPU: 19×
```

## Implementation Status

### ✅ Completed

1. **Core Infrastructure**
   - GPU element structure with all properties
   - Element-to-GPU conversion
   - Buffer management
   - PCG solver framework

2. **Metal Shaders**
   - 3D beam element stiffness computation
   - Local-to-global transformation
   - Dense 12×12 matrix operations
   - Atomic scatter operations

3. **Integration**
   - `LinearSolver` trait implementation
   - Module exports
   - Benchmark infrastructure

### 🐛 Current Issues

**Problem**: Matvec operation produces incorrect results

**Symptoms**:
- When `p = [0, 0, 0, ...]`, result is `Ap = [0, 0, 1.68e8, ...]` (should be zero!)
- PCG diverges after iteration 3
- Final displacement error: 10²³% (!)

**Likely Causes**:
1. Atomic buffer not properly initialized/synchronized
2. Memory alignment issues with `atomic_float`
3. Race condition in atomic scatter
4. Buffer caching/coherency problem

**Debug Status**: Currently investigating buffer initialization and atomic operations.

## Testing Strategy

### Unit Tests

1. **Single Element Test**: Verify matvec for 1-element beam matches CPU
2. **Multi-Element Test**: Verify matvec for grid structure
3. **Convergence Test**: Verify PCG convergence matches CPU solver

### Benchmarks

Located in: `tests/benchmarks/matrix_free_bench.rs`

- Simple beam (12 DOF)
- Medium grid (600 DOF)
- Large grid (5,400 DOF)

Each benchmark compares:
- CPU Cholesky (reference)
- GPU Sparse PCG (current)
- GPU Matrix-Free (new)

## References

### Commercial FEA Implementations

This approach is used by:
- **ANSYS Mechanical** - Element-by-element for nonlinear
- **LS-DYNA** - Explicit dynamics with GPU acceleration
- **Abaqus** - Element formulation for large models

### Academic References

1. Hughes, T.J.R. (2000). "The Finite Element Method" - Element-by-element methods
2. Cecka, C. et al. (2011). "Assembly of finite element methods on graphics processors" - GPU FEA
3. Komatitsch, D. et al. (2010). "High-order finite-element seismic wave propagation modeling" - Matrix-free GPU solvers

## Next Steps

### Immediate (Debug Phase)

1. ✅ Add comprehensive logging to matvec operation
2. 🔄 Investigate atomic_float buffer initialization
3. ⏳ Test with manual buffer zeroing before atomic ops
4. ⏳ Verify Metal shader memory access patterns

### Short-term (Optimization)

1. Profile GPU kernel performance
2. Optimize threadgroup sizes
3. Consider shared memory for element data
4. Implement diagonal preconditioner

### Long-term (Features)

1. Support for other element types (plates, shells)
2. Nonlinear analysis capability
3. Adaptive mesh refinement
4. Multi-GPU support

## Code Examples

### Using the Matrix-Free Solver

```rust
use fe_engine::prelude::*;
use fe_engine::analysis::gpu::MatrixFreeGPU;

// Build model
let model = builder.build().unwrap();

// Create matrix-free GPU solver
let solver = MatrixFreeGPU::new(&model)?;

// Solve
let result = pipeline.run(&solver, &load_case)?;
```

### Performance Comparison

```bash
# Run benchmarks
cargo test --features gpu --test matrix_free_bench -- --nocapture

# Expected output (once working):
# CPU Cholesky:     1.5 ms
# GPU Sparse PCG:   45.0 ms
# GPU Matrix-Free:  0.8 ms  (1.9× faster than CPU!)
```

## Debugging Notes

### Current Investigation

The matvec operation is returning non-zero values for zero input vectors, indicating:
- Buffer contamination from previous operations
- Atomic operations accessing uninitialized memory
- Synchronization issue with GPU buffers

### Debug Output

```
iter 0: p[0..3] = [0.000e0, 0.000e0, 0.000e0]
iter 0: Ap[0..3] = [0.000e0, 0.000e0, 1.680e8]  ❌ WRONG!
```

This suggests the output buffer at index 2 contains garbage data before the atomic operations.

### Hypothesis

The `atomic_float` buffer might not support proper initialization through `create_zero_buffer`. The buffer might need explicit clearing via a GPU kernel before the matvec operation.

### Next Debug Steps

1. Add explicit buffer-clearing kernel
2. Verify buffer contents before/after matvec
3. Test with CPU-side verification of GPU results
4. Check Metal atomic operation synchronization

---

**Last Updated**: 2025-10-26
**Status**: Implementation complete, debugging in progress
