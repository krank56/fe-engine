pub mod assembler;
pub mod cpu_solver;
pub mod elements;
pub mod error;
pub mod gpu;
pub mod pipeline;
pub mod result;
pub mod solver;

pub use assembler::GlobalAssembler;
pub use cpu_solver::CpuCholesky;
pub use error::SolverError;
pub use pipeline::AnalysisPipeline;
pub use result::{
    AnalysisResult, ElementForceComponents, ElementForces, ModelSnapshot, NodalDisplacement,
    SolverBackend, SolverInfo, SupportId, SupportReaction,
};
pub use solver::{LinearSolver, auto_select_solver, create_solver, is_gpu_available};
