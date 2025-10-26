#include <metal_stdlib>
using namespace metal;

// Helper function for parallel reduction (handles non-power-of-2 threadgroup sizes)
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

kernel void spmv_kernel(
    const device float* values [[buffer(0)]],
    const device int* col_indices [[buffer(1)]],
    const device int* row_offsets [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const device uint* n [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= n[0]) {
        return;
    }
    
    int row_start = row_offsets[row];
    int row_end = row_offsets[row + 1];
    
    float sum = 0.0;
    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * x[col_indices[j]];
    }
    
    y[row] = sum;
}

// Parallel reduction for dot product using threadgroup memory
kernel void dot_product_to_buffer_kernel(
    const device float* x [[buffer(0)]],
    const device float* y [[buffer(1)]],
    device float* result [[buffer(2)]],
    const device uint* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    // Load and compute product (all threads must participate)
    float local_sum = 0.0f;
    if (gid < n[0]) {
        local_sum = x[gid] * y[gid];
    }

    // Store to threadgroup memory and perform reduction
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    parallel_reduce_sum(shared_data, tid, tg_size);

    // First thread in each threadgroup writes result
    if (tid == 0) {
        atomic_fetch_add_explicit((device atomic_float*)&result[0], shared_data[0], memory_order_relaxed);
    }
}

kernel void dot_product_kernel(
    const device float* x [[buffer(0)]],
    const device float* y [[buffer(1)]],
    device float* partial_sum [[buffer(2)]],
    const device uint* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    // Load and compute product
    float local_sum = 0.0f;
    if (gid < n[0]) {
        local_sum = x[gid] * y[gid];
    }

    // Store to threadgroup memory and perform reduction
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    parallel_reduce_sum(shared_data, tid, tg_size);

    // First thread in each threadgroup writes result
    if (tid == 0) {
        atomic_fetch_add_explicit((device atomic_float*)&partial_sum[0], shared_data[0], memory_order_relaxed);
    }
}

kernel void pcg_iteration_kernel(
    const device float* values [[buffer(0)]],
    const device int* col_indices [[buffer(1)]],
    const device int* row_offsets [[buffer(2)]],
    const device float* diag [[buffer(3)]],
    device float* x [[buffer(4)]],
    device float* r [[buffer(5)]],
    device float* z [[buffer(6)]],
    device float* p [[buffer(7)]],
    device float* ap [[buffer(8)]],
    device float* dot_rz [[buffer(9)]],
    device float* dot_pap [[buffer(10)]],
    device float* dot_rz_new [[buffer(11)]],
    device float* residual_norm [[buffer(12)]],
    const device uint* n [[buffer(13)]],
    const device uint* stage [[buffer(14)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint idx = gid;
    uint n_val = n[0];
    uint stage_val = stage[0];

    if (stage_val == 0) {
        // SpMV: ap = A * p
        if (idx < n_val) {
            int row_start = row_offsets[idx];
            int row_end = row_offsets[idx + 1];
            float sum = 0.0;
            for (int j = row_start; j < row_end; j++) {
                sum += values[j] * p[col_indices[j]];
            }
            ap[idx] = sum;
        }
    }
    else if (stage_val == 1) {
        // Dot product: p · ap with parallel reduction
        float local_sum = 0.0f;
        if (idx < n_val) {
            local_sum = p[idx] * ap[idx];
        }

        shared_data[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        parallel_reduce_sum(shared_data, tid, tg_size);

        if (tid == 0) {
            atomic_fetch_add_explicit((device atomic_float*)&dot_pap[0], shared_data[0], memory_order_relaxed);
        }
    }
    else if (stage_val == 2) {
        // Dot product: r · z with parallel reduction
        float local_sum = 0.0f;
        if (idx < n_val) {
            local_sum = r[idx] * z[idx];
        }

        shared_data[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        parallel_reduce_sum(shared_data, tid, tg_size);

        if (tid == 0) {
            atomic_fetch_add_explicit((device atomic_float*)&dot_rz_new[0], shared_data[0], memory_order_relaxed);
        }
    }
    else if (stage_val == 3) {
        // Norm: r · r with parallel reduction
        float local_sum = 0.0f;
        if (idx < n_val) {
            float val = r[idx];
            local_sum = val * val;
        }

        shared_data[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        parallel_reduce_sum(shared_data, tid, tg_size);

        if (tid == 0) {
            atomic_fetch_add_explicit((device atomic_float*)&residual_norm[0], shared_data[0], memory_order_relaxed);
        }
    }
}

kernel void pcg_update_vectors_kernel(
    const device float* alpha [[buffer(0)]],
    const device float* beta [[buffer(1)]],
    const device float* diag [[buffer(2)]],
    device float* x [[buffer(3)]],
    device float* r [[buffer(4)]],
    device float* z [[buffer(5)]],
    device float* p [[buffer(6)]],
    const device float* ap [[buffer(7)]],
    const device uint* n [[buffer(8)]],
    const device uint* update_stage [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid;
    uint n_val = n[0];
    uint stage = update_stage[0];
    
    if (idx >= n_val) {
        return;
    }
    
    if (stage == 0) {
        x[idx] += alpha[0] * p[idx];
        r[idx] -= alpha[0] * ap[idx];
    }
    else if (stage == 1) {
        z[idx] = r[idx] / diag[idx];
    }
    else if (stage == 2) {
        p[idx] = z[idx] + beta[0] * p[idx];
    }
}

kernel void compute_alpha_beta_kernel(
    const device float* dot_rz [[buffer(0)]],
    const device float* dot_pap [[buffer(1)]],
    const device float* dot_rz_new [[buffer(2)]],
    device float* alpha [[buffer(3)]],
    device float* beta [[buffer(4)]],
    device float* rz_old_out [[buffer(5)]],
    const device uint* stage [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    uint s = stage[0];
    
    if (s == 0) {
        float rz_old = dot_rz[0];
        float pap = dot_pap[0];
        if (pap > 1e-20f) {
            alpha[0] = rz_old / pap;
        } else {
            alpha[0] = 0.0f;
        }
        rz_old_out[0] = rz_old;
    }
    else if (s == 1) {
        float rz_old = rz_old_out[0];
        float rz_new = dot_rz_new[0];
        if (rz_old > 1e-20f) {
            beta[0] = rz_new / rz_old;
        } else {
            beta[0] = 0.0f;
        }
    }
}

kernel void axpy_kernel(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* alpha [[buffer(2)]],
    const device uint* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n[0]) {
        return;
    }
    y[gid] += alpha[0] * x[gid];
}

kernel void precondition_kernel(
    const device float* diag [[buffer(0)]],
    const device float* r [[buffer(1)]],
    device float* z [[buffer(2)]],
    const device uint* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n[0]) {
        return;
    }
    z[gid] = r[gid] / diag[gid];
}

kernel void vector_update_kernel(
    device float* p [[buffer(0)]],
    const device float* z [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device uint* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n[0]) {
        return;
    }
    p[gid] = z[gid] + beta[0] * p[gid];
}

kernel void norm2_kernel(
    const device float* x [[buffer(0)]],
    device float* partial_sum [[buffer(1)]],
    const device uint* n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    // Load and compute squared value
    float local_sum = 0.0f;
    if (gid < n[0]) {
        float val = x[gid];
        local_sum = val * val;
    }

    // Store to threadgroup memory and perform reduction
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    parallel_reduce_sum(shared_data, tid, tg_size);

    // First thread in each threadgroup writes result
    if (tid == 0) {
        atomic_fetch_add_explicit((device atomic_float*)&partial_sum[0], shared_data[0], memory_order_relaxed);
    }
}

kernel void zero_scalar_kernel(
    device float* buffer [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        buffer[0] = 0.0f;
    }
}

kernel void copy_scalar_kernel(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        dst[0] = src[0];
    }
}

kernel void copy_kernel(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device uint* n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n[0]) {
        return;
    }
    dst[gid] = src[gid];
}
