use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

use crate::analysis::solver::LinearSolver;
use crate::analysis::SolverError;

#[cfg(feature = "gpu")]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions};

/// GPU-accelerated PCG solver using Metal
///
/// **Status**: EXPERIMENTAL - Currently 8-10× slower than CPU due to:
/// - Atomic operations for dot products (should use parallel reduction)
/// - Multiple kernel dispatches per iteration (~8-10 pipeline switches)
/// - Small workload sizes (<20K DOF don't amortize GPU overhead)
///
/// **Accuracy**: Results match CPU solver within <0.3% error
///
/// **Recommendation**: Use CPU solver (Cholesky) for models <20K DOF.
/// GPU solver infrastructure is in place for future optimization.
#[cfg(feature = "gpu")]
pub struct MetalPCG {
    device: Device,
    command_queue: CommandQueue,
    pipeline_spmv: ComputePipelineState,
    pipeline_dot: ComputePipelineState,
    pipeline_axpy: ComputePipelineState,
    pipeline_precondition: ComputePipelineState,
    pipeline_vector_update: ComputePipelineState,
    pipeline_norm2: ComputePipelineState,
    pipeline_copy: ComputePipelineState,
    pipeline_copy_scalar: ComputePipelineState,
    pipeline_zero_scalar: ComputePipelineState,
    pipeline_pcg_iteration: ComputePipelineState,
    pipeline_pcg_update_vectors: ComputePipelineState,
    pipeline_dot_to_buffer: ComputePipelineState,
    pipeline_compute_alpha_beta: ComputePipelineState,
}

#[cfg(not(feature = "gpu"))]
pub struct MetalPCG {}

impl MetalPCG {
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "gpu")]
        {
            let device = Device::system_default()
                .ok_or_else(|| "No Metal-compatible GPU found".to_string())?;

            let command_queue = device.new_command_queue();

            let library = Self::compile_shaders(&device)?;

            let pipeline_spmv = Self::create_pipeline(&device, &library, "spmv_kernel")?;
            let pipeline_dot = Self::create_pipeline(&device, &library, "dot_product_kernel")?;
            let pipeline_axpy = Self::create_pipeline(&device, &library, "axpy_kernel")?;
            let pipeline_precondition =
                Self::create_pipeline(&device, &library, "precondition_kernel")?;
            let pipeline_vector_update =
                Self::create_pipeline(&device, &library, "vector_update_kernel")?;
            let pipeline_norm2 = Self::create_pipeline(&device, &library, "norm2_kernel")?;
            let pipeline_copy = Self::create_pipeline(&device, &library, "copy_kernel")?;
            let pipeline_copy_scalar =
                Self::create_pipeline(&device, &library, "copy_scalar_kernel")?;
            let pipeline_zero_scalar =
                Self::create_pipeline(&device, &library, "zero_scalar_kernel")?;
            let pipeline_pcg_iteration =
                Self::create_pipeline(&device, &library, "pcg_iteration_kernel")?;
            let pipeline_pcg_update_vectors =
                Self::create_pipeline(&device, &library, "pcg_update_vectors_kernel")?;
            let pipeline_dot_to_buffer =
                Self::create_pipeline(&device, &library, "dot_product_to_buffer_kernel")?;
            let pipeline_compute_alpha_beta =
                Self::create_pipeline(&device, &library, "compute_alpha_beta_kernel")?;

            Ok(Self {
                device,
                command_queue,
                pipeline_spmv,
                pipeline_dot,
                pipeline_axpy,
                pipeline_precondition,
                pipeline_vector_update,
                pipeline_norm2,
                pipeline_copy,
                pipeline_copy_scalar,
                pipeline_zero_scalar,
                pipeline_pcg_iteration,
                pipeline_pcg_update_vectors,
                pipeline_dot_to_buffer,
                pipeline_compute_alpha_beta,
            })
        }

        #[cfg(not(feature = "gpu"))]
        {
            Err("GPU support not enabled (compile with --features gpu)".to_string())
        }
    }

    #[cfg(feature = "gpu")]
    fn compile_shaders(device: &Device) -> Result<Library, String> {
        let shader_source = include_str!("shaders/pcg.metal");

        device
            .new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| format!("Failed to compile Metal shaders: {}", e))
    }

    #[cfg(feature = "gpu")]
    fn create_pipeline(
        device: &Device,
        library: &Library,
        function_name: &str,
    ) -> Result<ComputePipelineState, String> {
        let function = library
            .get_function(function_name, None)
            .map_err(|e| format!("Failed to get function '{}': {}", function_name, e))?;

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline for '{}': {}", function_name, e))
    }

    #[cfg(feature = "gpu")]
    fn create_buffer<T>(&self, data: &[T]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);

        unsafe {
            let ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        buffer
    }

    #[cfg(feature = "gpu")]
    fn create_buffer_f32_from_f64(&self, data: &[f64]) -> Buffer {
        let float_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        self.create_buffer(&float_data)
    }

    #[cfg(feature = "gpu")]
    fn create_zero_buffer(&self, size: usize) -> Buffer {
        let data = vec![0.0f32; size];
        self.create_buffer(&data)
    }

    #[cfg(feature = "gpu")]
    fn gpu_pcg_solve_resident(
        &self,
        k: &CsrMatrix<f64>,
        f: &DVector<f64>,
    ) -> Result<DVector<f64>, SolverError> {
        let n = k.nrows();

        let values: Vec<f64> = k.values().to_vec();
        let col_indices: Vec<i32> = k.col_indices().iter().map(|&i| i as i32).collect();
        let row_offsets: Vec<i32> = k.row_offsets().iter().map(|&i| i as i32).collect();

        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let row_start = k.row_offsets()[i];
                let row_end = k.row_offsets()[i + 1];
                for j in row_start..row_end {
                    if k.col_indices()[j] == i {
                        return k.values()[j];
                    }
                }
                1.0
            })
            .collect();

        let buf_values = self.create_buffer_f32_from_f64(&values);
        let buf_col_indices = self.create_buffer(&col_indices);
        let buf_row_offsets = self.create_buffer(&row_offsets);
        let buf_diag = self.create_buffer_f32_from_f64(&diag);

        let x = vec![0.0; n];
        let r = f.as_slice().to_vec();

        let buf_x = self.create_buffer_f32_from_f64(&x);
        let buf_r = self.create_buffer_f32_from_f64(&r);
        let buf_z = self.create_zero_buffer(n);
        let buf_p = self.create_zero_buffer(n);
        let buf_ap = self.create_zero_buffer(n);

        let buf_alpha = self.create_zero_buffer(1);
        let buf_beta = self.create_zero_buffer(1);
        let buf_dot_rz = self.create_zero_buffer(1);
        let buf_dot_pap = self.create_zero_buffer(1);
        let buf_dot_rz_new = self.create_zero_buffer(1);
        let buf_residual_norm = self.create_zero_buffer(1);
        let buf_rz_old = self.create_zero_buffer(1);

        let buf_n = self.create_buffer(&[n as u32]);
        let buf_stage0 = self.create_buffer(&[0u32]);
        let buf_stage1 = self.create_buffer(&[1u32]);
        let buf_stage2 = self.create_buffer(&[2u32]);

        let max_iterations = n.max(1000);
        let tolerance = 1e-5f32;

        // Use fixed 256 threadgroup size (power of 2 for efficient reduction)
        let threadgroup_width = 256u64;
        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(threadgroup_width, 1, 1);
        let threadgroup_mem_size = (threadgroup_width * std::mem::size_of::<f32>() as u64) as u64;

        // Initialization phase
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_precondition);
        encoder.set_buffer(0, Some(&buf_diag), 0);
        encoder.set_buffer(1, Some(&buf_r), 0);
        encoder.set_buffer(2, Some(&buf_z), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.set_compute_pipeline_state(&self.pipeline_copy);
        encoder.set_buffer(0, Some(&buf_z), 0);
        encoder.set_buffer(1, Some(&buf_p), 0);
        encoder.set_buffer(2, Some(&buf_n), 0);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.set_compute_pipeline_state(&self.pipeline_dot_to_buffer);
        encoder.set_buffer(0, Some(&buf_r), 0);
        encoder.set_buffer(1, Some(&buf_z), 0);
        encoder.set_buffer(2, Some(&buf_dot_rz), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Main PCG iteration loop - batch iterations to reduce synchronization
        let batch_size = 10;
        for batch_start in (0..max_iterations).step_by(batch_size) {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let single_thread = metal::MTLSize::new(1, 1, 1);

            for _iter_in_batch in 0..batch_size.min(max_iterations - batch_start) {
                // Zero reduction buffers
                encoder.set_compute_pipeline_state(&self.pipeline_zero_scalar);
                encoder.set_buffer(0, Some(&buf_dot_pap), 0);
                encoder.dispatch_threads(single_thread, single_thread);

                encoder.set_compute_pipeline_state(&self.pipeline_zero_scalar);
                encoder.set_buffer(0, Some(&buf_dot_rz_new), 0);
                encoder.dispatch_threads(single_thread, single_thread);

                // SpMV: ap = A * p
                encoder.set_compute_pipeline_state(&self.pipeline_spmv);
                encoder.set_buffer(0, Some(&buf_values), 0);
                encoder.set_buffer(1, Some(&buf_col_indices), 0);
                encoder.set_buffer(2, Some(&buf_row_offsets), 0);
                encoder.set_buffer(3, Some(&buf_p), 0);
                encoder.set_buffer(4, Some(&buf_ap), 0);
                encoder.set_buffer(5, Some(&buf_n), 0);
                encoder.dispatch_threads(grid_size, threadgroup_size);

                // Dot product: p · ap (with parallel reduction)
                encoder.set_compute_pipeline_state(&self.pipeline_dot_to_buffer);
                encoder.set_buffer(0, Some(&buf_p), 0);
                encoder.set_buffer(1, Some(&buf_ap), 0);
                encoder.set_buffer(2, Some(&buf_dot_pap), 0);
                encoder.set_buffer(3, Some(&buf_n), 0);
                encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);
                encoder.dispatch_threads(grid_size, threadgroup_size);

                // Compute alpha
                encoder.set_compute_pipeline_state(&self.pipeline_compute_alpha_beta);
                encoder.set_buffer(0, Some(&buf_dot_rz), 0);
                encoder.set_buffer(1, Some(&buf_dot_pap), 0);
                encoder.set_buffer(2, Some(&buf_dot_rz_new), 0);
                encoder.set_buffer(3, Some(&buf_alpha), 0);
                encoder.set_buffer(4, Some(&buf_beta), 0);
                encoder.set_buffer(5, Some(&buf_rz_old), 0);
                encoder.set_buffer(6, Some(&buf_stage0), 0);
                encoder.dispatch_threads(single_thread, single_thread);

                // Update x and r
                encoder.set_compute_pipeline_state(&self.pipeline_pcg_update_vectors);
                encoder.set_buffer(0, Some(&buf_alpha), 0);
                encoder.set_buffer(1, Some(&buf_beta), 0);
                encoder.set_buffer(2, Some(&buf_diag), 0);
                encoder.set_buffer(3, Some(&buf_x), 0);
                encoder.set_buffer(4, Some(&buf_r), 0);
                encoder.set_buffer(5, Some(&buf_z), 0);
                encoder.set_buffer(6, Some(&buf_p), 0);
                encoder.set_buffer(7, Some(&buf_ap), 0);
                encoder.set_buffer(8, Some(&buf_n), 0);
                encoder.set_buffer(9, Some(&buf_stage0), 0);
                encoder.dispatch_threads(grid_size, threadgroup_size);

                // Precondition: z = M^-1 * r
                encoder.set_compute_pipeline_state(&self.pipeline_pcg_update_vectors);
                encoder.set_buffer(0, Some(&buf_alpha), 0);
                encoder.set_buffer(1, Some(&buf_beta), 0);
                encoder.set_buffer(2, Some(&buf_diag), 0);
                encoder.set_buffer(3, Some(&buf_x), 0);
                encoder.set_buffer(4, Some(&buf_r), 0);
                encoder.set_buffer(5, Some(&buf_z), 0);
                encoder.set_buffer(6, Some(&buf_p), 0);
                encoder.set_buffer(7, Some(&buf_ap), 0);
                encoder.set_buffer(8, Some(&buf_n), 0);
                encoder.set_buffer(9, Some(&buf_stage1), 0);
                encoder.dispatch_threads(grid_size, threadgroup_size);

                // Dot product: r · z (with parallel reduction)
                encoder.set_compute_pipeline_state(&self.pipeline_dot_to_buffer);
                encoder.set_buffer(0, Some(&buf_r), 0);
                encoder.set_buffer(1, Some(&buf_z), 0);
                encoder.set_buffer(2, Some(&buf_dot_rz_new), 0);
                encoder.set_buffer(3, Some(&buf_n), 0);
                encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);
                encoder.dispatch_threads(grid_size, threadgroup_size);

                // Compute beta
                encoder.set_compute_pipeline_state(&self.pipeline_compute_alpha_beta);
                encoder.set_buffer(0, Some(&buf_dot_rz), 0);
                encoder.set_buffer(1, Some(&buf_dot_pap), 0);
                encoder.set_buffer(2, Some(&buf_dot_rz_new), 0);
                encoder.set_buffer(3, Some(&buf_alpha), 0);
                encoder.set_buffer(4, Some(&buf_beta), 0);
                encoder.set_buffer(5, Some(&buf_rz_old), 0);
                encoder.set_buffer(6, Some(&buf_stage1), 0);
                encoder.dispatch_threads(single_thread, single_thread);

                // Update p
                encoder.set_compute_pipeline_state(&self.pipeline_pcg_update_vectors);
                encoder.set_buffer(0, Some(&buf_alpha), 0);
                encoder.set_buffer(1, Some(&buf_beta), 0);
                encoder.set_buffer(2, Some(&buf_diag), 0);
                encoder.set_buffer(3, Some(&buf_x), 0);
                encoder.set_buffer(4, Some(&buf_r), 0);
                encoder.set_buffer(5, Some(&buf_z), 0);
                encoder.set_buffer(6, Some(&buf_p), 0);
                encoder.set_buffer(7, Some(&buf_ap), 0);
                encoder.set_buffer(8, Some(&buf_n), 0);
                encoder.set_buffer(9, Some(&buf_stage2), 0);
                encoder.dispatch_threads(grid_size, threadgroup_size);

                // Copy rz_new to rz
                encoder.set_compute_pipeline_state(&self.pipeline_copy_scalar);
                encoder.set_buffer(0, Some(&buf_dot_rz_new), 0);
                encoder.set_buffer(1, Some(&buf_dot_rz), 0);
                encoder.dispatch_threads(single_thread, single_thread);
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check convergence less frequently (every batch instead of every 100 iters)
            let command_buffer2 = self.command_queue.new_command_buffer();
            let encoder2 = command_buffer2.new_compute_command_encoder();

            encoder2.set_compute_pipeline_state(&self.pipeline_zero_scalar);
            encoder2.set_buffer(0, Some(&buf_residual_norm), 0);
            let single_thread = metal::MTLSize::new(1, 1, 1);
            encoder2.dispatch_threads(single_thread, single_thread);

            encoder2.set_compute_pipeline_state(&self.pipeline_dot_to_buffer);
            encoder2.set_buffer(0, Some(&buf_r), 0);
            encoder2.set_buffer(1, Some(&buf_r), 0);
            encoder2.set_buffer(2, Some(&buf_residual_norm), 0);
            encoder2.set_buffer(3, Some(&buf_n), 0);
            encoder2.set_threadgroup_memory_length(0, threadgroup_mem_size);
            encoder2.dispatch_threads(grid_size, threadgroup_size);

            encoder2.end_encoding();
            command_buffer2.commit();
            command_buffer2.wait_until_completed();

            let residual_norm = unsafe {
                let ptr = buf_residual_norm.contents() as *const f32;
                (*ptr).sqrt()
            };

            if residual_norm < tolerance {
                let mut x_result = vec![0.0; n];
                self.copy_buffer_f32_to_vec_f64(&buf_x, &mut x_result, n);
                return Ok(DVector::from_vec(x_result));
            }
        }

        let mut x_result = vec![0.0; n];
        self.copy_buffer_f32_to_vec_f64(&buf_x, &mut x_result, n);
        Ok(DVector::from_vec(x_result))
    }

    #[cfg(feature = "gpu")]
    fn gpu_pcg_solve(
        &self,
        k: &CsrMatrix<f64>,
        f: &DVector<f64>,
    ) -> Result<DVector<f64>, SolverError> {
        let n = k.nrows();

        let values: Vec<f64> = k.values().to_vec();
        let col_indices: Vec<i32> = k.col_indices().iter().map(|&i| i as i32).collect();
        let row_offsets: Vec<i32> = k.row_offsets().iter().map(|&i| i as i32).collect();

        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let row_start = k.row_offsets()[i];
                let row_end = k.row_offsets()[i + 1];
                for j in row_start..row_end {
                    if k.col_indices()[j] == i {
                        return k.values()[j];
                    }
                }
                1.0
            })
            .collect();

        let buf_values = self.create_buffer_f32_from_f64(&values);
        let buf_col_indices = self.create_buffer(&col_indices);
        let buf_row_offsets = self.create_buffer(&row_offsets);
        let buf_diag = self.create_buffer_f32_from_f64(&diag);

        let x = vec![0.0; n];
        let r = f.as_slice().to_vec();
        let z = vec![0.0; n];
        let p = vec![0.0; n];
        let ap = vec![0.0; n];

        let buf_x = self.create_buffer_f32_from_f64(&x);
        let buf_r = self.create_buffer_f32_from_f64(&r);
        let buf_z = self.create_buffer_f32_from_f64(&z);
        let buf_p = self.create_buffer_f32_from_f64(&p);
        let buf_ap = self.create_buffer_f32_from_f64(&ap);

        let max_iterations = n.max(1000);
        let tolerance = 1e-5;

        self.precondition_gpu(&buf_diag, &buf_r, &buf_z, n)?;
        self.copy_gpu(&buf_z, &buf_p, n)?;

        let mut rz_old = self.dot_product_gpu(&buf_r, &buf_z, n)?;

        for iter in 0..max_iterations {
            self.spmv_gpu(
                &buf_values,
                &buf_col_indices,
                &buf_row_offsets,
                &buf_p,
                &buf_ap,
                n,
            )?;

            let p_ap = self.dot_product_gpu(&buf_p, &buf_ap, n)?;
            let alpha = rz_old / p_ap;

            let alpha_buf = self.create_buffer(&[alpha as f32]);
            self.axpy_gpu(&buf_x, &buf_p, &alpha_buf, n)?;

            let neg_alpha_buf = self.create_buffer(&[-(alpha as f32)]);
            self.axpy_gpu(&buf_r, &buf_ap, &neg_alpha_buf, n)?;

            let residual_norm = self.norm2_gpu(&buf_r, n)?;

            if residual_norm < tolerance {
                let mut x_result = vec![0.0; n];
                self.copy_buffer_f32_to_vec_f64(&buf_x, &mut x_result, n);
                return Ok(DVector::from_vec(x_result));
            }

            self.precondition_gpu(&buf_diag, &buf_r, &buf_z, n)?;

            let rz_new = self.dot_product_gpu(&buf_r, &buf_z, n)?;
            let beta = rz_new / rz_old;

            let beta_buf = self.create_buffer(&[beta as f32]);
            self.vector_update_gpu(&buf_p, &buf_z, &beta_buf, n)?;

            rz_old = rz_new;
        }

        let mut x_result = vec![0.0; n];
        self.copy_buffer_f32_to_vec_f64(&buf_x, &mut x_result, n);
        Ok(DVector::from_vec(x_result))
    }

    #[cfg(feature = "gpu")]
    fn spmv_gpu(
        &self,
        values: &Buffer,
        col_indices: &Buffer,
        row_offsets: &Buffer,
        x: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<(), SolverError> {
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_spmv);
        encoder.set_buffer(0, Some(values), 0);
        encoder.set_buffer(1, Some(col_indices), 0);
        encoder.set_buffer(2, Some(row_offsets), 0);
        encoder.set_buffer(3, Some(x), 0);
        encoder.set_buffer(4, Some(y), 0);
        encoder.set_buffer(5, Some(&buf_n), 0);

        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn dot_product_gpu(&self, x: &Buffer, y: &Buffer, n: usize) -> Result<f64, SolverError> {
        let partial_sum = vec![0.0f32; 1];
        let buf_partial = self.create_buffer(&partial_sum);
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_dot);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(y), 0);
        encoder.set_buffer(2, Some(&buf_partial), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);

        // Use fixed 256 threadgroup size (power of 2 for efficient reduction)
        let threadgroup_width = 256u64;
        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(threadgroup_width, 1, 1);
        let threadgroup_mem_size = (threadgroup_width * std::mem::size_of::<f32>() as u64) as u64;

        encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut result = vec![0.0f32; 1];
        unsafe {
            let ptr = buf_partial.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), 1);
        }

        Ok(result[0] as f64)
    }

    #[cfg(feature = "gpu")]
    fn axpy_gpu(
        &self,
        y: &Buffer,
        x: &Buffer,
        alpha: &Buffer,
        n: usize,
    ) -> Result<(), SolverError> {
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_axpy);
        encoder.set_buffer(0, Some(y), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(alpha), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);

        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn precondition_gpu(
        &self,
        diag: &Buffer,
        r: &Buffer,
        z: &Buffer,
        n: usize,
    ) -> Result<(), SolverError> {
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_precondition);
        encoder.set_buffer(0, Some(diag), 0);
        encoder.set_buffer(1, Some(r), 0);
        encoder.set_buffer(2, Some(z), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);

        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn vector_update_gpu(
        &self,
        p: &Buffer,
        z: &Buffer,
        beta: &Buffer,
        n: usize,
    ) -> Result<(), SolverError> {
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_vector_update);
        encoder.set_buffer(0, Some(p), 0);
        encoder.set_buffer(1, Some(z), 0);
        encoder.set_buffer(2, Some(beta), 0);
        encoder.set_buffer(3, Some(&buf_n), 0);

        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn norm2_gpu(&self, x: &Buffer, n: usize) -> Result<f64, SolverError> {
        let partial_sum = vec![0.0f32; 1];
        let buf_partial = self.create_buffer(&partial_sum);
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_norm2);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(&buf_partial), 0);
        encoder.set_buffer(2, Some(&buf_n), 0);

        // Use fixed 256 threadgroup size (power of 2 for efficient reduction)
        let threadgroup_width = 256u64;
        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(threadgroup_width, 1, 1);
        let threadgroup_mem_size = (threadgroup_width * std::mem::size_of::<f32>() as u64) as u64;

        encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut result = vec![0.0f32; 1];
        unsafe {
            let ptr = buf_partial.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), 1);
        }

        Ok((result[0] as f64).sqrt())
    }

    #[cfg(feature = "gpu")]
    fn copy_gpu(&self, src: &Buffer, dst: &Buffer, n: usize) -> Result<(), SolverError> {
        let buf_n = self.create_buffer(&[n as u32]);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_copy);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), 0);
        encoder.set_buffer(2, Some(&buf_n), 0);

        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn copy_buffer_f32_to_vec_f64(&self, buffer: &Buffer, vec: &mut [f64], n: usize) {
        let mut temp = vec![0.0f32; n];
        unsafe {
            let ptr = buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, temp.as_mut_ptr(), n);
        }
        for i in 0..n {
            vec[i] = temp[i] as f64;
        }
    }
}

impl LinearSolver for MetalPCG {
    fn solve(
        &self,
        k: &CsrMatrix<f64>,
        f: &DVector<f64>,
        _bc: Option<&crate::analysis::solver::BoundaryConditions>,
    ) -> Result<DVector<f64>, SolverError> {
        // GPU Sparse PCG uses elimination strategy
        // BC parameter is ignored - pipeline already reduced the system

        #[cfg(feature = "gpu")]
        {
            // Use optimized batched version
            self.gpu_pcg_solve_resident(k, f)
        }

        #[cfg(not(feature = "gpu"))]
        {
            let _ = (k, f);
            Err(SolverError::NotImplemented(
                "GPU support not enabled".to_string(),
            ))
        }
    }

    fn name(&self) -> &str {
        "Metal PCG (GPU-Optimized)"
    }

    // Uses default BcStrategy::Elimination
}
