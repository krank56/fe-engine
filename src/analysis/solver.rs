use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

use crate::analysis::cpu_solver::CpuCholesky;
use crate::analysis::SolverError;
use crate::structure::support::Support;

#[cfg(all(target_os = "macos", feature = "gpu"))]
use crate::analysis::gpu::MetalPCG;

/// Strategy for handling boundary conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcStrategy {
    /// Traditional approach: eliminate constrained DOFs from system
    /// Used by: CPU Cholesky, GPU Sparse PCG
    Elimination,

    /// Penalty method: keep all DOFs, enforce constraints with large penalties
    /// Used by: GPU Matrix-Free solvers
    Penalty,
}

/// Boundary condition information for solvers
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    /// List of supports in the model
    pub supports: Vec<Support>,

    /// Total number of DOFs in the full system (before BC application)
    pub total_dofs: usize,

    /// Indices of constrained DOFs in the full system
    pub constrained_dofs: Vec<usize>,

    /// Indices of free DOFs in the full system
    pub free_dofs: Vec<usize>,
}

impl BoundaryConditions {
    /// Create boundary conditions from model supports
    pub fn from_supports(supports: Vec<Support>, total_dofs: usize) -> Self {
        let mut constrained_dofs = Vec::new();

        for support in &supports {
            let base_dof = support.node_id * 6;

            // Determine which DOFs are constrained based on support type
            use crate::structure::support::SupportType;
            match support.support_type {
                SupportType::Fixed => {
                    // All 6 DOFs constrained
                    for i in 0..6 {
                        constrained_dofs.push(base_dof + i);
                    }
                }
                SupportType::Pinned => {
                    // Translations constrained (0, 1, 2), rotations free
                    for i in 0..3 {
                        constrained_dofs.push(base_dof + i);
                    }
                }
                SupportType::Roller { .. } => {
                    // For now, treat as pinned (translations constrained)
                    // TODO: Handle free_direction properly
                    for i in 0..3 {
                        constrained_dofs.push(base_dof + i);
                    }
                }
                SupportType::ElasticSpring { .. } => {
                    // TODO: Handle elastic springs properly
                    // For now, treat as fixed
                    for i in 0..6 {
                        constrained_dofs.push(base_dof + i);
                    }
                }
            }
        }

        // Sort and deduplicate
        constrained_dofs.sort_unstable();
        constrained_dofs.dedup();

        // Build free DOFs list (all DOFs not in constrained list)
        let free_dofs: Vec<usize> = (0..total_dofs)
            .filter(|dof| !constrained_dofs.contains(dof))
            .collect();

        Self {
            supports,
            total_dofs,
            constrained_dofs,
            free_dofs,
        }
    }
}

pub trait LinearSolver {
    /// Solve the linear system K·u = f
    ///
    /// # Parameters
    /// - `k`: Stiffness matrix (may be reduced or full depending on solver strategy)
    /// - `f`: Force vector (may be reduced or full depending on solver strategy)
    /// - `bc`: Optional boundary condition information (used by matrix-free solvers)
    ///
    /// # Returns
    /// Solution vector u (size matches f)
    fn solve(
        &self,
        k: &CsrMatrix<f64>,
        f: &DVector<f64>,
        bc: Option<&BoundaryConditions>,
    ) -> Result<DVector<f64>, SolverError>;

    fn name(&self) -> &str;

    /// Return the boundary condition strategy used by this solver
    fn boundary_condition_strategy(&self) -> BcStrategy {
        BcStrategy::Elimination  // Default: traditional approach
    }
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
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        const GPU_THRESHOLD: usize = 5000;
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
