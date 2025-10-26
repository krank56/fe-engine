use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    SingularMatrix { message: String },
    NumericalInstability { message: String },
    ConvergenceFailure { iterations: usize, residual: f64 },
    InvalidInput { message: String },
    OutOfMemory { required_bytes: usize },
    UnsupportedConfiguration { message: String },
    FactorizationError(String),
    SolutionError(String),
    NotImplemented(String),
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::SingularMatrix { message } => {
                write!(f, "Singular stiffness matrix: {}", message)
            }
            SolverError::NumericalInstability { message } => {
                write!(f, "Numerical instability: {}", message)
            }
            SolverError::ConvergenceFailure {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Solver failed to converge after {} iterations (residual: {:.2e})",
                    iterations, residual
                )
            }
            SolverError::InvalidInput { message } => {
                write!(f, "Invalid solver input: {}", message)
            }
            SolverError::OutOfMemory { required_bytes } => {
                write!(
                    f,
                    "Out of memory: requires {} MB",
                    required_bytes / (1024 * 1024)
                )
            }
            SolverError::UnsupportedConfiguration { message } => {
                write!(f, "Unsupported configuration: {}", message)
            }
            SolverError::FactorizationError(message) => {
                write!(f, "Matrix factorization failed: {}", message)
            }
            SolverError::SolutionError(message) => {
                write!(f, "Linear system solution failed: {}", message)
            }
            SolverError::NotImplemented(message) => {
                write!(f, "Not implemented: {}", message)
            }
        }
    }
}

impl std::error::Error for SolverError {}
