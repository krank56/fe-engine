use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

use crate::analysis::SolverError;
use crate::analysis::cpu_solver::CpuCholesky;

#[cfg(all(target_os = "macos", feature = "gpu"))]
use crate::analysis::gpu::MetalPCG;

pub trait LinearSolver {
    fn solve(
        &self,
        k: &CsrMatrix<f64>,
        f: &DVector<f64>,
    ) -> Result<DVector<f64>, SolverError>;

    fn name(&self) -> &str;
}

pub fn create_solver(backend: crate::analysis::result::SolverBackend) -> Box<dyn LinearSolver> {
    match backend {
        crate::analysis::result::SolverBackend::CpuCholesky => Box::new(CpuCholesky),
        crate::analysis::result::SolverBackend::GpuIterative => {
            #[cfg(all(target_os = "macos", feature = "gpu"))]
            {
                match MetalPCG::new() {
                    Ok(gpu_solver) => Box::new(gpu_solver),
                    Err(_e) => Box::new(CpuCholesky),
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "gpu")))]
            {
                Box::new(CpuCholesky)
            }
        }
        _ => Box::new(CpuCholesky),
    }
}

pub fn auto_select_solver(num_dofs: usize) -> crate::analysis::result::SolverBackend {
    const GPU_THRESHOLD: usize = 5000;
    
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        if num_dofs >= GPU_THRESHOLD && MetalPCG::new().is_ok() {
            return crate::analysis::result::SolverBackend::GpuIterative;
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        let _ = num_dofs;
    }
    
    crate::analysis::result::SolverBackend::CpuCholesky
}

pub fn is_gpu_available() -> bool {
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        MetalPCG::new().is_ok()
    }
    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        false
    }
}
