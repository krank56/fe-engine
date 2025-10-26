#include <metal_stdlib>
using namespace metal;

/// Clear atomic buffer to zero
/// CRITICAL: atomic_float buffers may not be properly initialized by default
kernel void clear_buffer(
    device atomic_float* buffer [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
) {
    atomic_store_explicit(&buffer[idx], 0.0f, memory_order_relaxed);
}

/// GPU-optimized beam element structure
struct GpuBeamElement {
    // Material properties
    float E;   // Young's modulus
    float G;   // Shear modulus
    float A;   // Cross-sectional area
    float Iy;  // Moment of inertia y
    float Iz;  // Moment of inertia z
    float J;   // Torsional constant

    // Geometry
    float length;

    // Direction cosines for coordinate transformation
    float cos_x;
    float cos_y;
    float cos_z;

    // DOF mapping (12 DOFs total: 6 per node × 2 nodes)
    uint dofs[12];
};

/// Compute 3D beam element stiffness matrix in local coordinates
/// This produces a dense 12×12 matrix - GPU loves dense operations!
void compute_beam_stiffness_local(
    GpuBeamElement e,
    thread float* k
) {
    float L = e.length;
    float E = e.E;
    float A = e.A;
    float Iy = e.Iy;
    float Iz = e.Iz;
    float G = e.G;
    float J = e.J;

    // Precompute common terms
    float EA_L = E * A / L;
    float GJ_L = G * J / L;

    // Bending stiffness about y-axis (in x-z plane)
    float EIy_L = E * Iy / L;
    float EIy_L2 = EIy_L / L;
    float EIy_L3 = EIy_L2 / L;
    float c12y = 12.0f * EIy_L3;
    float c6y = 6.0f * EIy_L2;
    float c4y = 4.0f * EIy_L;
    float c2y = 2.0f * EIy_L;

    // Bending stiffness about z-axis (in x-y plane)
    float EIz_L = E * Iz / L;
    float EIz_L2 = EIz_L / L;
    float EIz_L3 = EIz_L2 / L;
    float c12z = 12.0f * EIz_L3;
    float c6z = 6.0f * EIz_L2;
    float c4z = 4.0f * EIz_L;
    float c2z = 2.0f * EIz_L;

    // Initialize matrix to zero
    for (uint i = 0; i < 144; i++) {
        k[i] = 0.0f;
    }

    // Axial stiffness (DOF 0 and 6: u_x at nodes i and j)
    k[0*12 + 0] = EA_L;
    k[0*12 + 6] = -EA_L;
    k[6*12 + 0] = -EA_L;
    k[6*12 + 6] = EA_L;

    // Bending in x-z plane (DOF 2, 4, 8, 10: u_z and θ_y)
    k[2*12 + 2] = c12y;
    k[2*12 + 4] = -c6y;
    k[2*12 + 8] = -c12y;
    k[2*12 + 10] = -c6y;

    k[4*12 + 2] = -c6y;
    k[4*12 + 4] = c4y;
    k[4*12 + 8] = c6y;
    k[4*12 + 10] = c2y;

    k[8*12 + 2] = -c12y;
    k[8*12 + 4] = c6y;
    k[8*12 + 8] = c12y;
    k[8*12 + 10] = c6y;

    k[10*12 + 2] = -c6y;
    k[10*12 + 4] = c2y;
    k[10*12 + 8] = c6y;
    k[10*12 + 10] = c4y;

    // Bending in x-y plane (DOF 1, 5, 7, 11: u_y and θ_z)
    k[1*12 + 1] = c12z;
    k[1*12 + 5] = c6z;
    k[1*12 + 7] = -c12z;
    k[1*12 + 11] = c6z;

    k[5*12 + 1] = c6z;
    k[5*12 + 5] = c4z;
    k[5*12 + 7] = -c6z;
    k[5*12 + 11] = c2z;

    k[7*12 + 1] = -c12z;
    k[7*12 + 5] = -c6z;
    k[7*12 + 7] = c12z;
    k[7*12 + 11] = -c6z;

    k[11*12 + 1] = c6z;
    k[11*12 + 5] = c2z;
    k[11*12 + 7] = -c6z;
    k[11*12 + 11] = c4z;

    // Torsion (DOF 3 and 9: θ_x)
    k[3*12 + 3] = GJ_L;
    k[3*12 + 9] = -GJ_L;
    k[9*12 + 3] = -GJ_L;
    k[9*12 + 9] = GJ_L;
}

/// Compute transformation matrix from local to global coordinates
void compute_transformation_matrix(
    float cx, float cy, float cz,
    thread float* T
) {
    // Initialize to zero
    for (uint i = 0; i < 144; i++) {
        T[i] = 0.0f;
    }

    // The transformation matrix is block diagonal: T = [R 0; 0 R; 0 0; 0 0]
    // where R is the 3×3 rotation matrix, repeated 4 times for the 4 sets of 3 DOFs

    // Handle special case where element is aligned with x-axis
    float tolerance = 1e-6f;
    float lambda_matrix[9];

    if (abs(cx - 1.0f) < tolerance && abs(cy) < tolerance && abs(cz) < tolerance) {
        // Element aligned with x-axis
        lambda_matrix[0] = 1.0f; lambda_matrix[1] = 0.0f; lambda_matrix[2] = 0.0f;
        lambda_matrix[3] = 0.0f; lambda_matrix[4] = 1.0f; lambda_matrix[5] = 0.0f;
        lambda_matrix[6] = 0.0f; lambda_matrix[7] = 0.0f; lambda_matrix[8] = 1.0f;
    } else {
        // General case: construct rotation matrix
        // Local x-axis aligned with element (cx, cy, cz)
        // Local y and z axes perpendicular to local x

        float D = sqrt(cx*cx + cz*cz);

        if (D > tolerance) {
            // Standard case
            lambda_matrix[0] = cx;
            lambda_matrix[1] = cy;
            lambda_matrix[2] = cz;

            lambda_matrix[3] = -cy * cz / D;
            lambda_matrix[4] = D;
            lambda_matrix[5] = -cy * cx / D;

            lambda_matrix[6] = -cx / D;
            lambda_matrix[7] = 0.0f;
            lambda_matrix[8] = cz / D;
        } else {
            // Element nearly vertical
            lambda_matrix[0] = cx;
            lambda_matrix[1] = cy;
            lambda_matrix[2] = cz;

            lambda_matrix[3] = 0.0f;
            lambda_matrix[4] = 0.0f;
            lambda_matrix[5] = 1.0f;

            lambda_matrix[6] = -1.0f;
            lambda_matrix[7] = 0.0f;
            lambda_matrix[8] = 0.0f;
        }
    }

    // Fill the 12×12 transformation matrix (block diagonal)
    for (uint block = 0; block < 4; block++) {
        for (uint i = 0; i < 3; i++) {
            for (uint j = 0; j < 3; j++) {
                T[(block*3 + i)*12 + (block*3 + j)] = lambda_matrix[i*3 + j];
            }
        }
    }
}

/// Dense 12×12 matrix-vector multiply
/// This is what GPUs are designed for!
void matvec_12x12(
    const thread float* A,
    const thread float* x,
    thread float* y
) {
    for (uint i = 0; i < 12; i++) {
        float sum = 0.0f;
        for (uint j = 0; j < 12; j++) {
            sum += A[i*12 + j] * x[j];
        }
        y[i] = sum;
    }
}

/// Main kernel: Each thread processes ONE element
/// This achieves perfect parallelism - all elements computed simultaneously!
kernel void element_matvec_kernel(
    const device GpuBeamElement* elements [[buffer(0)]],
    const device float* v_global [[buffer(1)]],
    device atomic_float* result [[buffer(2)]],
    uint elem_id [[thread_position_in_grid]]
) {
    // Thread-private memory for intermediate computations
    // Each thread needs its own copies - NOT threadgroup shared!
    float k_local[144];      // 12×12 element stiffness (local coords)
    float T[144];            // 12×12 transformation matrix
    float k_global[144];     // 12×12 element stiffness (global coords)
    float v_local[12];       // Element DOFs from global vector
    float f_local[12];       // Element forces (local)

    // Get element data
    GpuBeamElement e = elements[elem_id];

    // Step 1: Extract element DOFs from global vector
    for (uint i = 0; i < 12; i++) {
        v_local[i] = v_global[e.dofs[i]];
    }

    // Step 2: Compute element stiffness matrix in local coordinates
    // This is a dense 12×12 matrix - GPU loves this!
    compute_beam_stiffness_local(e, k_local);

    // Step 3: Transform to global coordinates: k_global = T^T × k_local × T
    compute_transformation_matrix(e.cos_x, e.cos_y, e.cos_z, T);

    // First: temp = k_local × T
    float temp_matrix[144];
    for (uint i = 0; i < 12; i++) {
        for (uint j = 0; j < 12; j++) {
            float sum = 0.0f;
            for (uint k = 0; k < 12; k++) {
                sum += k_local[i*12 + k] * T[k*12 + j];
            }
            temp_matrix[i*12 + j] = sum;
        }
    }

    // Second: k_global = T^T × temp_matrix
    for (uint i = 0; i < 12; i++) {
        for (uint j = 0; j < 12; j++) {
            float sum = 0.0f;
            for (uint k = 0; k < 12; k++) {
                sum += T[k*12 + i] * temp_matrix[k*12 + j];  // T^T
            }
            k_global[i*12 + j] = sum;
        }
    }

    // Step 4: Compute element forces: f_local = k_global × v_local
    // Dense matrix-vector multiply - GPU optimized!
    matvec_12x12(k_global, v_local, f_local);

    // Step 5: Scatter element forces back to global result vector
    // Use atomic add to handle contributions from multiple elements to same DOF
    for (uint i = 0; i < 12; i++) {
        atomic_fetch_add_explicit(&result[e.dofs[i]], f_local[i], memory_order_relaxed);
    }
}
