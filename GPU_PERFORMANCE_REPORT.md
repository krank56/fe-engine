# GPU Solver Performance Report

## Test Date
Generated: 2025-10-26

## System Configuration
- Platform: macOS (Metal API)
- GPU: Apple Silicon / AMD GPU
- CPU Solver: Sparse Cholesky (nalgebra-sparse)
- GPU Solver: Preconditioned Conjugate Gradient (Metal Compute)

## Benchmark Results

### Small Models (< 100 DOF)

#### Simple Beam (2 nodes, 1 element, 12 DOF)
```
CPU Cholesky: 0.100 ms
Metal PCG:    84.378 ms
Speedup:      0.00x (GPU is 843× slower)
Accuracy:     3.889e-9 m error (< 0.001%)
```

#### Portal Frame (4 nodes, 3 elements, 12 DOF)
```
CPU Cholesky: 0.135 ms
Metal PCG:    127.508 ms
Speedup:      0.00x (GPU is 944× slower)
Accuracy:     6.575e-9 m error (< 0.001%)
```

### Medium Models (100-1000 DOF)

#### Grid 10×10 (100 nodes, 180 elements, 600 DOF)
```
CPU Cholesky: 4.365 ms
Metal PCG:    127.945 ms
Speedup:      0.03x (GPU is 29× slower)
Accuracy:     8.640e-5 m error (< 0.01%)
```

## Analysis

### Current Performance Characteristics

**GPU Overhead Breakdown:**
1. **Shader Compilation**: ~20-30ms (first-time compilation)
2. **Data Transfer (CPU→GPU)**: Matrix values, indices, vectors
3. **Kernel Dispatch**: 10-12 dispatches per PCG iteration
4. **Synchronization**: Command buffer commits every 10 iterations
5. **Data Transfer (GPU→CPU)**: Result vector

**Why GPU is Slower for Small Models:**

The GPU solver has fixed overhead costs that dominate for small problems:
- Shader compilation and pipeline setup: ~20-30ms
- CPU-GPU data transfer latency: ~5-10ms per transfer
- Kernel launch overhead: ~0.1-0.5ms per dispatch
- For a 12 DOF problem that solves in 0.1ms on CPU, GPU overhead is 800-1000× the compute time

### Optimizations Implemented

✅ **Completed Optimizations:**
1. Replaced atomic operations with parallel reduction (threadgroup memory)
2. Batched 10 PCG iterations per command buffer (10× fewer synchronization points)
3. Reduced convergence check frequency (every 10 iterations vs every 100)
4. Fixed numerical correctness for non-power-of-2 threadgroup sizes

### Remaining Bottlenecks

⚠️ **Critical Issues:**
1. **Kernel Dispatch Overhead**: Still ~10 kernel dispatches per iteration
   - SpMV, dot products, vector updates are separate kernels
   - Each dispatch has ~0.1-0.5ms overhead

2. **CPU-GPU Transfer**: Data must be transferred at beginning and end
   - Matrix data (values, indices, offsets): Depends on sparsity
   - Vector data (x, r, p, z): 4 × N × sizeof(f32) bytes

3. **Iterative vs Direct**: PCG typically needs 10-1000 iterations
   - CPU Cholesky: Direct solver, 1 factorization + 2 triangular solves
   - GPU PCG: Iterative, needs many iterations to converge

### Performance Projection

**Estimated Crossover Point:**
Based on overhead analysis, GPU solver would need models with:
- **Minimum ~50,000 DOF** to amortize fixed overhead
- **Dense matrices** where SpMV dominates (not typical for FEA)
- **Poor conditioning** where Cholesky struggles but PCG works

**For Typical FEA Models:**
- Sparse matrices (< 1% non-zeros)
- Well-conditioned systems (E modulus variations < 100×)
- CPU Cholesky is optimal for DOF < 100,000

## Recommendations

### For Production Use
**✅ Use CPU Cholesky Solver**
- Faster for all tested problem sizes
- Direct solver (guaranteed convergence)
- Well-tested and stable
- Handles ill-conditioned systems better

### For GPU Solver Development
**Potential Improvements (would require significant work):**

1. **Kernel Fusion** (High Impact)
   - Combine SpMV + dot product + vector update into single kernel
   - Could reduce 10 dispatches → 2-3 dispatches per iteration
   - Estimated improvement: 3-5× speedup

2. **Persistent Thread Approach** (Very High Impact)
   - Keep threads alive across iterations
   - Eliminate kernel launch overhead entirely
   - Estimated improvement: 5-10× speedup

3. **Better Preconditioner** (Medium Impact)
   - Current: Jacobi (diagonal only)
   - Upgrade to: Incomplete Cholesky or multigrid
   - Could reduce iterations by 5-10×

4. **Mixed Precision** (Low Impact)
   - Already using f32 on GPU
   - Could try f16 for some operations
   - Small benefit for memory bandwidth

## Conclusion

**Current Status:**
The GPU solver is **numerically correct** but **not performance-competitive** for typical FEA problems. The optimizations implemented (parallel reduction, batching) improve efficiency but cannot overcome the fundamental overhead issues.

**Recommendation:**
- Continue using CPU Cholesky solver for production
- GPU solver remains experimental research code
- Would need months of optimization work to achieve parity, let alone speedup

**GPU Solver is Best For:**
- Research and learning Metal compute programming
- Very large, dense linear systems (not typical in FEA)
- Specialized applications where iterative methods excel

## Technical Details

### Implemented Optimizations

**1. Parallel Reduction (src/analysis/gpu/shaders/pcg.metal:4-21)**
```metal
inline void parallel_reduce_sum(threadgroup float* shared_data, uint tid, uint tg_size) {
    // Find largest power of 2 <= tg_size
    uint s = 1;
    while (s < tg_size) {
        s <<= 1;
    }
    s >>= 1;

    // Reduction loop
    while (s > 0) {
        if (tid < s && tid + s < tg_size) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s >>= 1;
    }
}
```

Benefits:
- Reduces N atomic operations → log₂(256) operations per threadgroup
- Eliminates GPU memory contention
- Proper handling of non-power-of-2 sizes

**2. Iteration Batching (src/analysis/gpu/metal.rs:241-393)**
```rust
let batch_size = 10;
for batch_start in (0..max_iterations).step_by(batch_size) {
    let command_buffer = self.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    for _iter_in_batch in 0..batch_size.min(max_iterations - batch_start) {
        // All 10 iterations recorded into single command buffer
    }

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();  // Sync every 10 iterations
}
```

Benefits:
- Reduces CPU-GPU synchronization by 10×
- Better GPU utilization (less idle time)
- Lower command buffer overhead

## Accuracy Validation

All tests pass with error < 0.01%:
- ✅ Simple beam: 3.889e-9 m error
- ✅ Portal frame: 6.575e-9 m error
- ✅ Grid 10×10: 8.640e-5 m error

The GPU solver produces numerically correct results within engineering tolerance.
