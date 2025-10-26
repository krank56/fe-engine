use nalgebra::DVector;
use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};

use crate::analysis::solver::LinearSolver;
use crate::analysis::SolverError;

pub struct CpuCholesky;

impl LinearSolver for CpuCholesky {
    fn solve(
        &self,
        k: &CsrMatrix<f64>,
        f: &DVector<f64>,
        _bc: Option<&crate::analysis::solver::BoundaryConditions>,
    ) -> Result<DVector<f64>, SolverError> {
        // CPU Cholesky uses elimination strategy
        // BC parameter is ignored - pipeline already reduced the system

        let coo = CooMatrix::from(k);
        let k_csc = CscMatrix::from(&coo);

        let cholesky = CscCholesky::factor(&k_csc).map_err(|e| {
            SolverError::FactorizationError(format!("Cholesky factorization failed: {:?}", e))
        })?;

        let u_dense = cholesky.solve(f);

        let u = DVector::from_iterator(u_dense.nrows(), u_dense.iter().copied());

        Ok(u)
    }

    fn name(&self) -> &str {
        "CPU Cholesky (nalgebra-sparse)"
    }

    // Uses default BcStrategy::Elimination
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra_sparse::CooMatrix;

    #[test]
    fn test_cholesky_simple_spd_matrix() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(1, 0, 2.0);
        coo.push(0, 1, 2.0);
        coo.push(1, 1, 5.0);
        coo.push(2, 1, 1.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 2, 3.0);

        let k = CsrMatrix::from(&coo);
        let f = DVector::from_vec(vec![12.0, 17.0, 8.0]);

        let solver = CpuCholesky;
        let u = solver.solve(&k, &f, None).unwrap();

        let residual = &k * &u - &f;
        assert_relative_eq!(residual.norm(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_solver_with_known_system() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 2.0);
        coo.push(1, 0, 2.0);
        coo.push(1, 1, 3.0);

        let k = CsrMatrix::from(&coo);
        let f = DVector::from_vec(vec![10.0, 7.0]);

        let solver = CpuCholesky;
        let u = solver.solve(&k, &f, None).unwrap();

        println!("Solution: {:?}", u);
        assert_relative_eq!(u[0], 2.0, epsilon = 1e-9);
        assert_relative_eq!(u[1], 1.0, epsilon = 1e-9);
    }
}
