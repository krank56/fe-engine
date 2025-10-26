use nalgebra::DVector;
use std::time::Instant;

use crate::analysis::assembler::GlobalAssembler;
use crate::analysis::error::SolverError;
use crate::analysis::result::{
    AnalysisResult, ModelSnapshot, NodalDisplacement, SolverBackend, SolverInfo,
};
use crate::analysis::solver::LinearSolver;
use crate::audit::trail::{AuditTrail, AuditValue};
use crate::structure::geometry::Vector3D;
use crate::structure::load::LoadCase;
use crate::structure::model::StructuralModel;

pub struct AnalysisPipeline<'a> {
    model: &'a StructuralModel,
    pub audit_trail: AuditTrail,
}

impl<'a> AnalysisPipeline<'a> {
    pub fn new(model: &'a StructuralModel) -> Self {
        Self {
            model,
            audit_trail: AuditTrail::new(),
        }
    }

    pub fn run<S: LinearSolver>(
        &mut self,
        solver: &S,
        load_case: &LoadCase,
    ) -> Result<AnalysisResult, SolverError> {
        self.audit_trail.append_action("start_analysis");

        self.audit_trail.append_with_details(
            "assemble_global_stiffness",
            vec![
                (
                    "num_elements".to_string(),
                    AuditValue::Integer(self.model.elements.len() as i64),
                ),
                (
                    "num_nodes".to_string(),
                    AuditValue::Integer(self.model.nodes.len() as i64),
                ),
            ],
        );
        let mut k = GlobalAssembler::assemble_global_stiffness(self.model);
        self.audit_trail.append_with_details(
            "stiffness_matrix_assembled",
            vec![
                (
                    "matrix_size".to_string(),
                    AuditValue::Integer(k.nrows() as i64),
                ),
                ("nnz".to_string(), AuditValue::Integer(k.nnz() as i64)),
            ],
        );

        let k_original = k.clone();

        self.audit_trail.append_action("assemble_load_vector");
        let mut f = GlobalAssembler::assemble_load_vector(self.model, load_case);
        self.audit_trail.append_with_details(
            "load_vector_assembled",
            vec![
                ("num_dofs".to_string(), AuditValue::Integer(f.len() as i64)),
                ("load_norm".to_string(), AuditValue::Float(f.norm())),
            ],
        );

        let f_original = f.clone();

        // Check solver's boundary condition strategy
        use crate::analysis::solver::BcStrategy;
        let bc_strategy = solver.boundary_condition_strategy();

        let result = match bc_strategy {
            BcStrategy::Elimination => {
                // Traditional approach: reduce system by eliminating constrained DOFs
                self.audit_trail.append_with_details(
                    "apply_boundary_conditions",
                    vec![
                        (
                            "num_supports".to_string(),
                            AuditValue::Integer(self.model.supports.len() as i64),
                        ),
                        ("strategy".to_string(), AuditValue::String("Elimination".to_string())),
                    ],
                );
                GlobalAssembler::apply_boundary_conditions(&mut k, &mut f, self.model);

                self.audit_trail.append_with_details(
                    "solve_linear_system",
                    vec![
                        (
                            "solver".to_string(),
                            AuditValue::String(solver.name().to_string()),
                        ),
                        ("dofs".to_string(), AuditValue::Integer(f.len() as i64)),
                    ],
                );
                let start = Instant::now();
                let u = solver.solve(&k, &f, None)?;
                let elapsed = start.elapsed();
                self.audit_trail.append_with_details(
                    "linear_system_solved",
                    vec![
                        (
                            "solve_time_ms".to_string(),
                            AuditValue::Float(elapsed.as_secs_f64() * 1000.0),
                        ),
                        ("displacement_norm".to_string(), AuditValue::Float(u.norm())),
                    ],
                );
                (u, elapsed)
            }

            BcStrategy::Penalty => {
                // Matrix-free approach: work with full system using penalty method
                self.audit_trail.append_with_details(
                    "prepare_boundary_conditions",
                    vec![
                        (
                            "num_supports".to_string(),
                            AuditValue::Integer(self.model.supports.len() as i64),
                        ),
                        ("strategy".to_string(), AuditValue::String("Penalty".to_string())),
                    ],
                );

                // Build BC info for solver
                use crate::analysis::solver::BoundaryConditions;
                let total_dofs = self.model.nodes.len() * 6;
                let bc = BoundaryConditions::from_supports(self.model.supports.clone(), total_dofs);

                // Penalty method: pass FULL system (not reduced)
                // Solver handles constraints internally via penalty method
                self.audit_trail.append_with_details(
                    "solve_linear_system",
                    vec![
                        (
                            "solver".to_string(),
                            AuditValue::String(solver.name().to_string()),
                        ),
                        ("total_dofs".to_string(), AuditValue::Integer(total_dofs as i64)),
                        ("constrained_dofs".to_string(), AuditValue::Integer(bc.constrained_dofs.len() as i64)),
                    ],
                );
                let start = Instant::now();
                let u = solver.solve(&k, &f, Some(&bc))?;
                let elapsed = start.elapsed();
                self.audit_trail.append_with_details(
                    "linear_system_solved",
                    vec![
                        (
                            "solve_time_ms".to_string(),
                            AuditValue::Float(elapsed.as_secs_f64() * 1000.0),
                        ),
                        ("displacement_norm".to_string(), AuditValue::Float(u.norm())),
                    ],
                );
                (u, elapsed)
            }
        };

        let (u, solve_time) = result;

        self.audit_trail.append_action("compute_element_forces");
        let element_forces = GlobalAssembler::recover_element_forces(self.model, &u);
        self.audit_trail.append_with_details(
            "element_forces_computed",
            vec![(
                "num_elements".to_string(),
                AuditValue::Integer(element_forces.len() as i64),
            )],
        );

        self.audit_trail.append_action("compute_support_reactions");
        let reactions =
            GlobalAssembler::compute_reactions(self.model, &k_original, &u, &f_original);
        self.audit_trail.append_with_details(
            "support_reactions_computed",
            vec![(
                "num_reactions".to_string(),
                AuditValue::Integer(reactions.len() as i64),
            )],
        );

        self.verify_equilibrium(&f_original, &reactions);

        let displacements = self.convert_to_nodal_displacements(&u);

        let solver_backend = match solver.name() {
            "CpuCholesky" => SolverBackend::CpuCholesky,
            "CpuLU" => SolverBackend::CpuLU,
            "CpuIterative" => SolverBackend::CpuIterative,
            _ => SolverBackend::CpuCholesky,
        };

        let solver_info = SolverInfo::new(solver_backend, solve_time);

        let model_hash = format!("{:x}", md5::compute(format!("{:?}", self.model)));
        let model_snapshot = ModelSnapshot::new(model_hash);

        self.audit_trail.append_action("analysis_complete");

        Ok(AnalysisResult::new(
            model_snapshot,
            displacements,
            element_forces,
            reactions,
            solver_info,
        ))
    }

    fn verify_equilibrium(
        &mut self,
        applied_loads: &DVector<f64>,
        reactions: &[crate::analysis::result::SupportReaction],
    ) {
        let total_applied_force = Vector3D {
            x: applied_loads.iter().step_by(6).sum(),
            y: applied_loads.iter().skip(1).step_by(6).sum(),
            z: applied_loads.iter().skip(2).step_by(6).sum(),
        };

        let total_reaction_force = reactions.iter().fold(Vector3D::zero(), |acc, r| Vector3D {
            x: acc.x + r.force.x,
            y: acc.y + r.force.y,
            z: acc.z + r.force.z,
        });

        let force_balance = Vector3D {
            x: total_applied_force.x + total_reaction_force.x,
            y: total_applied_force.y + total_reaction_force.y,
            z: total_applied_force.z + total_reaction_force.z,
        };

        let force_imbalance = force_balance.magnitude();

        self.audit_trail.append_with_details(
            "verify_equilibrium",
            vec![
                (
                    "force_imbalance".to_string(),
                    AuditValue::Float(force_imbalance),
                ),
                (
                    "equilibrium_satisfied".to_string(),
                    AuditValue::Boolean(force_imbalance <= 1e-6),
                ),
            ],
        );
    }

    fn convert_to_nodal_displacements(&self, u: &DVector<f64>) -> Vec<NodalDisplacement> {
        let mut displacements = Vec::new();

        for node in &self.model.nodes {
            let base_dof = node.id * 6;

            let translation = Vector3D {
                x: u[base_dof],
                y: u[base_dof + 1],
                z: u[base_dof + 2],
            };

            let rotation = Vector3D {
                x: u[base_dof + 3],
                y: u[base_dof + 4],
                z: u[base_dof + 5],
            };

            displacements.push(NodalDisplacement::new(node.id, translation, rotation));
        }

        displacements
    }
}
