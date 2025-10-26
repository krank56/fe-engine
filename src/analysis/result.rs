use crate::structure::element::ElementId;
use crate::structure::geometry::Vector3D;
use crate::structure::node::NodeId;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodalDisplacement {
    pub node_id: NodeId,
    pub translation: Vector3D,
    pub rotation: Vector3D,
}

impl NodalDisplacement {
    pub fn new(node_id: NodeId, translation: Vector3D, rotation: Vector3D) -> Self {
        Self {
            node_id,
            translation,
            rotation,
        }
    }

    pub fn total_displacement(&self) -> f64 {
        self.translation.magnitude()
    }

    pub fn total_rotation(&self) -> f64 {
        self.rotation.magnitude()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementForces {
    pub element_id: ElementId,
    pub forces: ElementForceComponents,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElementForceComponents {
    Beam {
        axial: Vec<f64>,
        shear_y: Vec<f64>,
        shear_z: Vec<f64>,
        moment_y: Vec<f64>,
        moment_z: Vec<f64>,
        torsion: Vec<f64>,
        evaluation_points: Vec<f64>,
    },
    Shell {
        membrane_forces: Vec<(f64, f64, f64)>,
        bending_moments: Vec<(f64, f64, f64)>,
        shear_forces: Vec<(f64, f64)>,
        evaluation_points: Vec<(f64, f64)>,
    },
    Solid {
        stress_tensor: Vec<[f64; 6]>,
        strain_tensor: Vec<[f64; 6]>,
        evaluation_points: Vec<(f64, f64, f64)>,
    },
}

impl ElementForces {
    pub fn max_moment(&self) -> Option<(f64, f64)> {
        match &self.forces {
            ElementForceComponents::Beam {
                moment_y,
                moment_z,
                evaluation_points,
                ..
            } => {
                let max_y = moment_y
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())?;
                let max_z = moment_z
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())?;

                let max_moment = max_y.1.abs().max(max_z.1.abs());
                let location = if max_y.1.abs() > max_z.1.abs() {
                    evaluation_points[max_y.0]
                } else {
                    evaluation_points[max_z.0]
                };

                Some((max_moment, location))
            }
            _ => None,
        }
    }

    pub fn max_shear(&self) -> Option<(f64, f64)> {
        match &self.forces {
            ElementForceComponents::Beam {
                shear_y,
                shear_z,
                evaluation_points,
                ..
            } => {
                let max_y = shear_y
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())?;
                let max_z = shear_z
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())?;

                let max_shear = max_y.1.abs().max(max_z.1.abs());
                let location = if max_y.1.abs() > max_z.1.abs() {
                    evaluation_points[max_y.0]
                } else {
                    evaluation_points[max_z.0]
                };

                Some((max_shear, location))
            }
            _ => None,
        }
    }
}

pub type SupportId = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportReaction {
    pub support_id: SupportId,
    pub node_id: NodeId,
    pub force: Vector3D,
    pub moment: Vector3D,
}

impl SupportReaction {
    pub fn new(support_id: SupportId, node_id: NodeId, force: Vector3D, moment: Vector3D) -> Self {
        Self {
            support_id,
            node_id,
            force,
            moment,
        }
    }

    pub fn total_force(&self) -> f64 {
        self.force.magnitude()
    }

    pub fn total_moment(&self) -> f64 {
        self.moment.magnitude()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverBackend {
    CpuCholesky,
    CpuLU,
    CpuIterative,
    GpuCholesky,
    GpuIterative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverInfo {
    pub solver_backend: SolverBackend,
    pub solve_time: Duration,
    pub iterations: Option<usize>,
    pub residual_norm: Option<f64>,
    pub software_version: String,
}

impl SolverInfo {
    pub fn new(solver_backend: SolverBackend, solve_time: Duration) -> Self {
        Self {
            solver_backend,
            solve_time,
            iterations: None,
            residual_norm: None,
            software_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn with_iterations(mut self, iterations: usize, residual_norm: f64) -> Self {
        self.iterations = Some(iterations);
        self.residual_norm = Some(residual_norm);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub model_hash: String,
    pub timestamp: String,
}

impl ModelSnapshot {
    pub fn new(model_hash: String) -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        Self {
            model_hash,
            timestamp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub model_snapshot: ModelSnapshot,
    pub displacements: Vec<NodalDisplacement>,
    pub element_forces: Vec<ElementForces>,
    pub reactions: Vec<SupportReaction>,
    pub solver_info: SolverInfo,
}

impl AnalysisResult {
    pub fn new(
        model_snapshot: ModelSnapshot,
        displacements: Vec<NodalDisplacement>,
        element_forces: Vec<ElementForces>,
        reactions: Vec<SupportReaction>,
        solver_info: SolverInfo,
    ) -> Self {
        Self {
            model_snapshot,
            displacements,
            element_forces,
            reactions,
            solver_info,
        }
    }

    pub fn displacement_at_node(&self, node_id: NodeId) -> Option<&NodalDisplacement> {
        self.displacements.iter().find(|d| d.node_id == node_id)
    }

    pub fn max_displacement(&self) -> f64 {
        self.displacements
            .iter()
            .map(|d| d.total_displacement())
            .fold(0.0, f64::max)
    }

    pub fn max_displacement_location(&self) -> (NodeId, Vector3D) {
        self.displacements
            .iter()
            .max_by(|a, b| {
                a.total_displacement()
                    .partial_cmp(&b.total_displacement())
                    .unwrap()
            })
            .map(|d| (d.node_id, d.translation))
            .unwrap_or((0, Vector3D::zero()))
    }

    pub fn displacements_in_direction(&self, direction: Vector3D) -> Vec<f64> {
        let dir_normalized = direction.normalize();
        self.displacements
            .iter()
            .map(|d| d.translation.dot(&dir_normalized))
            .collect()
    }

    pub fn forces_in_element(&self, element_id: ElementId) -> Option<&ElementForces> {
        self.element_forces
            .iter()
            .find(|f| f.element_id == element_id)
    }

    pub fn max_moment_in_element(&self, element_id: ElementId) -> Option<(f64, f64)> {
        self.forces_in_element(element_id)
            .and_then(|f| f.max_moment())
    }

    pub fn max_shear_in_element(&self, element_id: ElementId) -> Option<(f64, f64)> {
        self.forces_in_element(element_id)
            .and_then(|f| f.max_shear())
    }

    pub fn all_beam_forces(&self) -> Vec<&ElementForces> {
        self.element_forces
            .iter()
            .filter(|f| matches!(f.forces, ElementForceComponents::Beam { .. }))
            .collect()
    }

    pub fn total_reactions(&self) -> (Vector3D, Vector3D) {
        let total_force = self
            .reactions
            .iter()
            .fold(Vector3D::zero(), |acc, r| Vector3D {
                x: acc.x + r.force.x,
                y: acc.y + r.force.y,
                z: acc.z + r.force.z,
            });

        let total_moment = self
            .reactions
            .iter()
            .fold(Vector3D::zero(), |acc, r| Vector3D {
                x: acc.x + r.moment.x,
                y: acc.y + r.moment.y,
                z: acc.z + r.moment.z,
            });

        (total_force, total_moment)
    }

    pub fn reaction_at_support(&self, support_id: SupportId) -> Option<&SupportReaction> {
        self.reactions.iter().find(|r| r.support_id == support_id)
    }

    pub fn check_deflection_limit(&self, max_ratio: f64, span: f64) -> bool {
        let max_disp = self.max_displacement();
        max_disp <= span / max_ratio
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn verify_equilibrium(&self, applied_loads: &crate::structure::load::LoadCase, model: &crate::structure::model::StructuralModel) -> bool {
        use crate::structure::load::{Load, LoadDistribution};
        
        let (total_reactions, _) = self.total_reactions();
        
        let mut total_applied_force = Vector3D::zero();
        
        for load in &applied_loads.loads {
            match load {
                Load::NodalForce { force, .. } => {
                    total_applied_force.x += force.x;
                    total_applied_force.y += force.y;
                    total_applied_force.z += force.z;
                }
                Load::ElementLoad { element_id, load_distribution } => {
                    if let Some(element) = model.elements.iter().find(|e| e.id == *element_id) {
                        if let LoadDistribution::UniformDistributed { intensity, direction } = load_distribution {
                            let node_i = &model.nodes[element.connectivity[0]];
                            let node_j = &model.nodes[element.connectivity[1]];
                            let length = node_i.coordinates.distance_to(&node_j.coordinates);
                            
                            let total_load = *intensity * length;
                            total_applied_force.x += total_load * direction.x;
                            total_applied_force.y += total_load * direction.y;
                            total_applied_force.z += total_load * direction.z;
                        }
                    }
                }
                Load::ImposedDisplacement { .. } => {}
            }
        }
        
        let net_force_x = (total_reactions.x + total_applied_force.x).abs();
        let net_force_y = (total_reactions.y + total_applied_force.y).abs();
        let net_force_z = (total_reactions.z + total_applied_force.z).abs();
        
        let total_applied = total_applied_force.magnitude();
        let tolerance = total_applied.max(1.0) * 1e-6;
        
        net_force_x < tolerance && net_force_y < tolerance && net_force_z < tolerance
    }
}
