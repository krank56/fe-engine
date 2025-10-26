use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use std::collections::HashMap;

use crate::analysis::elements::beam::{BeamElement, BeamProperties};
use crate::analysis::elements::frame2d::Frame2DElement;
use crate::structure::element::{ElementType, Plane};
use crate::structure::load::{LoadCase, LoadDistribution};
use crate::structure::model::StructuralModel;
use crate::structure::support::{Direction, SupportType};

pub struct GlobalAssembler;

impl GlobalAssembler {
    pub fn assemble_global_stiffness(model: &StructuralModel) -> CsrMatrix<f64> {
        let num_nodes = model.nodes.len();
        let num_dof = num_nodes * 6;

        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

        let mut nodes_with_frame2d = std::collections::HashSet::new();
        let mut max_stiffness = 0.0_f64;

        for element in &model.elements {
            let material = &model.materials[element.material_id];
            let node_i = &model.nodes[element.connectivity[0]];
            let node_j = &model.nodes[element.connectivity[1]];

            match &element.element_type {
                ElementType::Frame2D { plane } => {
                    let frame = Frame2DElement::new(
                        &node_i.coordinates,
                        &node_j.coordinates,
                        material.elastic_modulus,
                        element.section.area,
                        match plane {
                            Plane::XY => element.section.inertia_z,
                            Plane::XZ => element.section.inertia_y,
                            Plane::YZ => element.section.inertia_z,
                        },
                        *plane,
                    );

                    let k_elem =
                        frame.global_stiffness_matrix(&node_i.coordinates, &node_j.coordinates);
                    let dof_map = frame.dof_mapping();

                    nodes_with_frame2d.insert(element.connectivity[0]);
                    nodes_with_frame2d.insert(element.connectivity[1]);

                    for i in 0..6 {
                        for j in 0..6 {
                            let val = k_elem[(i, j)].abs();
                            if val > max_stiffness {
                                max_stiffness = val;
                            }
                        }
                    }

                    let dof_i_base = element.connectivity[0] * 6;
                    let dof_j_base = element.connectivity[1] * 6;

                    for (local_i, &global_offset_i) in dof_map[0..3].iter().enumerate() {
                        for (local_j, &global_offset_j) in dof_map[0..3].iter().enumerate() {
                            let global_i = dof_i_base + global_offset_i;
                            let global_j = dof_i_base + global_offset_j;
                            triplets.push((global_i, global_j, k_elem[(local_i, local_j)]));
                        }
                    }

                    for (local_i, &global_offset_i) in dof_map[0..3].iter().enumerate() {
                        for (local_j, &global_offset_j) in dof_map[3..6].iter().enumerate() {
                            let global_i = dof_i_base + global_offset_i;
                            let global_j = dof_j_base + global_offset_j;
                            triplets.push((global_i, global_j, k_elem[(local_i, local_j + 3)]));
                        }
                    }

                    for (local_i, &global_offset_i) in dof_map[3..6].iter().enumerate() {
                        for (local_j, &global_offset_j) in dof_map[0..3].iter().enumerate() {
                            let global_i = dof_j_base + global_offset_i;
                            let global_j = dof_i_base + global_offset_j;
                            triplets.push((global_i, global_j, k_elem[(local_i + 3, local_j)]));
                        }
                    }

                    for (local_i, &global_offset_i) in dof_map[3..6].iter().enumerate() {
                        for (local_j, &global_offset_j) in dof_map[3..6].iter().enumerate() {
                            let global_i = dof_j_base + global_offset_i;
                            let global_j = dof_j_base + global_offset_j;
                            triplets.push((global_i, global_j, k_elem[(local_i + 3, local_j + 3)]));
                        }
                    }
                }
                ElementType::Beam1D { .. } | ElementType::Frame3D => {
                    let shear_modulus =
                        material.elastic_modulus / (2.0 * (1.0 + material.poisson_ratio));
                    let props = BeamProperties {
                        elastic_modulus: material.elastic_modulus,
                        shear_modulus,
                        area: element.section.area,
                        inertia_y: element.section.inertia_y,
                        inertia_z: element.section.inertia_z,
                        torsion_constant: element.section.torsion_constant,
                    };
                    let beam = BeamElement::new(&node_i.coordinates, &node_j.coordinates, &props);

                    let k_elem =
                        beam.global_stiffness_matrix(&node_i.coordinates, &node_j.coordinates);

                    for i in 0..12 {
                        for j in 0..12 {
                            let val = k_elem[(i, j)].abs();
                            if val > max_stiffness {
                                max_stiffness = val;
                            }
                        }
                    }

                    if element.id == 0 {
                        println!(
                            "Element 0: K_elem[2,2] (uz_i, uz_i) = {:.3e}",
                            k_elem[(2, 2)]
                        );
                        println!(
                            "Element 0: K_elem[2,4] (uz_i, θy_i) = {:.3e}",
                            k_elem[(2, 4)]
                        );
                        println!(
                            "Element 0: K_elem[4,4] (θy_i, θy_i) = {:.3e}",
                            k_elem[(4, 4)]
                        );
                        println!(
                            "Element 0: K_elem[2,10] (uz_i, θy_j) = {:.3e}",
                            k_elem[(2, 10)]
                        );
                    }

                    let dof_i = element.connectivity[0] * 6;
                    let dof_j = element.connectivity[1] * 6;

                    for i in 0..6 {
                        for j in 0..6 {
                            let global_i = dof_i + i;
                            let global_j = dof_i + j;
                            triplets.push((global_i, global_j, k_elem[(i, j)]));
                        }
                    }

                    for i in 0..6 {
                        for j in 0..6 {
                            let global_i = dof_i + i;
                            let global_j = dof_j + j;
                            triplets.push((global_i, global_j, k_elem[(i, j + 6)]));
                        }
                    }

                    for i in 0..6 {
                        for j in 0..6 {
                            let global_i = dof_j + i;
                            let global_j = dof_i + j;
                            triplets.push((global_i, global_j, k_elem[(i + 6, j)]));
                        }
                    }

                    for i in 0..6 {
                        for j in 0..6 {
                            let global_i = dof_j + i;
                            let global_j = dof_j + j;
                            triplets.push((global_i, global_j, k_elem[(i + 6, j + 6)]));
                        }
                    }
                }
                _ => {}
            }
        }

        if !nodes_with_frame2d.is_empty() && max_stiffness > 0.0 {
            let out_of_plane_penalty = max_stiffness * 1e-8;

            let mut frame2d_planes: std::collections::HashMap<usize, Plane> =
                std::collections::HashMap::new();

            for element in &model.elements {
                if let ElementType::Frame2D { plane } = &element.element_type {
                    for &node_id in &element.connectivity {
                        frame2d_planes.insert(node_id, *plane);
                    }
                }
            }

            for (&node_id, &plane) in &frame2d_planes {
                let dof_base = node_id * 6;

                let unconstrained_dofs: Vec<usize> = match plane {
                    Plane::XY => vec![2, 3, 4],
                    Plane::XZ => vec![1, 3, 5],
                    Plane::YZ => vec![0, 4, 5],
                };

                for &local_dof in &unconstrained_dofs {
                    let global_dof = dof_base + local_dof;
                    triplets.push((global_dof, global_dof, out_of_plane_penalty));
                }
            }
        }

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let mut map: HashMap<(usize, usize), f64> = HashMap::new();
        for (row, col, val) in triplets {
            *map.entry((row, col)).or_insert(0.0) += val;
        }

        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut current_row = 0;
        row_indices.push(0);

        for ((row, col), val) in entries {
            while current_row < row {
                current_row += 1;
                row_indices.push(col_indices.len());
            }
            col_indices.push(col);
            values.push(val);
        }

        while row_indices.len() <= num_dof {
            row_indices.push(col_indices.len());
        }

        CsrMatrix::try_from_csr_data(num_dof, num_dof, row_indices, col_indices, values)
            .expect("Invalid CSR matrix construction")
    }

    pub fn assemble_load_vector(model: &StructuralModel, load_case: &LoadCase) -> DVector<f64> {
        let num_dof = model.nodes.len() * 6;
        let mut f = DVector::zeros(num_dof);

        for load in &load_case.loads {
            match load {
                crate::structure::load::Load::NodalForce {
                    node_id,
                    force,
                    moment,
                } => {
                    let base_dof = *node_id * 6;
                    f[base_dof] += force.x;
                    f[base_dof + 1] += force.y;
                    f[base_dof + 2] += force.z;
                    f[base_dof + 3] += moment.x;
                    f[base_dof + 4] += moment.y;
                    f[base_dof + 5] += moment.z;
                }
                crate::structure::load::Load::ElementLoad {
                    element_id,
                    load_distribution:
                        LoadDistribution::UniformDistributed {
                            intensity,
                            direction,
                        },
                } => {
                    let element = &model.elements[*element_id];
                    let node_i_id = element.connectivity[0];
                    let node_j_id = element.connectivity[1];

                    let node_i = &model.nodes[node_i_id];
                    let node_j = &model.nodes[node_j_id];

                    let dx = node_j.coordinates.x - node_i.coordinates.x;
                    let dy = node_j.coordinates.y - node_i.coordinates.y;
                    let dz = node_j.coordinates.z - node_i.coordinates.z;
                    let length = (dx * dx + dy * dy + dz * dz).sqrt();

                    let total_force_x = intensity * direction.x * length;
                    let total_force_y = intensity * direction.y * length;
                    let total_force_z = intensity * direction.z * length;

                    let nodal_force_x = total_force_x / 2.0;
                    let nodal_force_y = total_force_y / 2.0;
                    let nodal_force_z = total_force_z / 2.0;

                    let fem_y = -(intensity * direction.z * length * length) / 12.0;
                    let fem_z = (intensity * direction.y * length * length) / 12.0;

                    let dof_i = node_i_id * 6;
                    let dof_j = node_j_id * 6;

                    f[dof_i] += nodal_force_x;
                    f[dof_i + 1] += nodal_force_y;
                    f[dof_i + 2] += nodal_force_z;
                    f[dof_i + 4] += fem_y;
                    f[dof_i + 5] += fem_z;

                    f[dof_j] += nodal_force_x;
                    f[dof_j + 1] += nodal_force_y;
                    f[dof_j + 2] += nodal_force_z;
                    f[dof_j + 4] -= fem_y;
                    f[dof_j + 5] -= fem_z;
                }
                _ => {}
            }
        }

        f
    }

    pub fn apply_boundary_conditions(
        k: &mut CsrMatrix<f64>,
        f: &mut DVector<f64>,
        model: &StructuralModel,
    ) {
        let penalty = 1e20;

        for support in &model.supports {
            let base_dof = support.node_id * 6;

            match &support.support_type {
                SupportType::Fixed => {
                    for i in 0..6 {
                        Self::apply_penalty_to_dof(k, f, base_dof + i, penalty);
                    }
                }
                SupportType::Pinned => {
                    for i in 0..3 {
                        Self::apply_penalty_to_dof(k, f, base_dof + i, penalty);
                    }
                }
                SupportType::Roller { free_direction } => {
                    // Roller: constrains translations perpendicular to rolling direction
                    // and torsion about the rolling axis, but frees bending rotations
                    for i in 0..3 {
                        let skip = match free_direction {
                            Direction::X => i == 0,
                            Direction::Y => i == 1,
                            Direction::Z => i == 2,
                        };
                        if !skip {
                            Self::apply_penalty_to_dof(k, f, base_dof + i, penalty);
                        }
                    }
                    // Constrain torsion about the free direction
                    let torsion_dof = match free_direction {
                        Direction::X => 3, // θx
                        Direction::Y => 4, // θy
                        Direction::Z => 5, // θz
                    };
                    Self::apply_penalty_to_dof(k, f, base_dof + torsion_dof, penalty);
                }
                SupportType::ElasticSpring { stiffness } => {
                    if let Some(kx) = stiffness.tx {
                        Self::apply_penalty_to_dof(k, f, base_dof, kx);
                    }
                    if let Some(ky) = stiffness.ty {
                        Self::apply_penalty_to_dof(k, f, base_dof + 1, ky);
                    }
                    if let Some(kz) = stiffness.tz {
                        Self::apply_penalty_to_dof(k, f, base_dof + 2, kz);
                    }
                    if let Some(krx) = stiffness.rx {
                        Self::apply_penalty_to_dof(k, f, base_dof + 3, krx);
                    }
                    if let Some(kry) = stiffness.ry {
                        Self::apply_penalty_to_dof(k, f, base_dof + 4, kry);
                    }
                    if let Some(krz) = stiffness.rz {
                        Self::apply_penalty_to_dof(k, f, base_dof + 5, krz);
                    }
                }
            }
        }
    }

    fn apply_penalty_to_dof(
        k: &mut CsrMatrix<f64>,
        _f: &mut DVector<f64>,
        dof: usize,
        penalty: f64,
    ) {
        let (row_offsets, col_indices, values) = k.csr_data_mut();

        let start = row_offsets[dof];
        let end = row_offsets[dof + 1];

        for idx in start..end {
            let col = col_indices[idx];
            if col == dof {
                values[idx] += penalty;
            }
        }
    }

    pub fn recover_element_forces(
        model: &StructuralModel,
        displacement_vector: &DVector<f64>,
    ) -> Vec<crate::analysis::result::ElementForces> {
        use crate::analysis::result::{ElementForceComponents, ElementForces};

        let mut element_forces = Vec::new();

        for element in &model.elements {
            match &element.element_type {
                ElementType::Beam1D { .. } | ElementType::Frame2D { .. } | ElementType::Frame3D => {
                    let material = &model.materials[element.material_id];
                    let node_i = &model.nodes[element.connectivity[0]];
                    let node_j = &model.nodes[element.connectivity[1]];

                    let shear_modulus =
                        material.elastic_modulus / (2.0 * (1.0 + material.poisson_ratio));
                    let props = BeamProperties {
                        elastic_modulus: material.elastic_modulus,
                        shear_modulus,
                        area: element.section.area,
                        inertia_y: element.section.inertia_y,
                        inertia_z: element.section.inertia_z,
                        torsion_constant: element.section.torsion_constant,
                    };
                    let beam = BeamElement::new(&node_i.coordinates, &node_j.coordinates, &props);

                    let dof_i = element.connectivity[0] * 6;
                    let dof_j = element.connectivity[1] * 6;

                    let mut u_global = nalgebra::DVector::zeros(12);
                    for i in 0..6 {
                        u_global[i] = displacement_vector[dof_i + i];
                        u_global[i + 6] = displacement_vector[dof_j + i];
                    }

                    let t = beam.transformation_matrix(&node_i.coordinates, &node_j.coordinates);

                    let u_local = t * u_global;

                    let k_local = beam.local_stiffness_matrix();

                    let f_local = k_local * u_local;

                    let axial_i = f_local[0];
                    let shear_y_i = f_local[2];
                    let shear_z_i = f_local[1];
                    let torsion_i = f_local[3];
                    let moment_y_i = f_local[4];
                    let moment_z_i = f_local[5];

                    let axial_j = f_local[6];
                    let shear_y_j = f_local[8];
                    let shear_z_j = f_local[7];
                    let torsion_j = f_local[9];
                    let moment_y_j = f_local[10];
                    let moment_z_j = f_local[11];

                    element_forces.push(ElementForces {
                        element_id: element.id,
                        forces: ElementForceComponents::Beam {
                            axial: vec![axial_i, axial_j],
                            shear_y: vec![shear_y_i, shear_y_j],
                            shear_z: vec![shear_z_i, shear_z_j],
                            moment_y: vec![moment_y_i, moment_y_j],
                            moment_z: vec![moment_z_i, moment_z_j],
                            torsion: vec![torsion_i, torsion_j],
                            evaluation_points: vec![0.0, 1.0],
                        },
                    });
                }
                _ => {}
            }
        }

        element_forces
    }

    pub fn compute_reactions(
        model: &StructuralModel,
        k: &CsrMatrix<f64>,
        displacement_vector: &DVector<f64>,
        applied_loads: &DVector<f64>,
    ) -> Vec<crate::analysis::result::SupportReaction> {
        use crate::analysis::result::SupportReaction;
        use crate::structure::geometry::Vector3D;

        let mut reactions = Vec::new();

        let f_internal = k * displacement_vector;

        for support in &model.supports {
            let base_dof = support.node_id * 6;

            let constrained_dofs = match &support.support_type {
                SupportType::Fixed => vec![0, 1, 2, 3, 4, 5],
                SupportType::Pinned => vec![0, 1, 2],
                SupportType::Roller { free_direction } => {
                    let mut dofs = vec![0, 1, 2];
                    let free_translation_dof = match free_direction {
                        Direction::X => 0,
                        Direction::Y => 1,
                        Direction::Z => 2,
                    };
                    dofs.retain(|&d| d != free_translation_dof);

                    let torsion_dof = match free_direction {
                        Direction::X => 3,
                        Direction::Y => 4,
                        Direction::Z => 5,
                    };
                    dofs.push(torsion_dof);

                    dofs
                }
                SupportType::ElasticSpring { stiffness } => {
                    let mut dofs = Vec::new();
                    if stiffness.tx.is_some() {
                        dofs.push(0);
                    }
                    if stiffness.ty.is_some() {
                        dofs.push(1);
                    }
                    if stiffness.tz.is_some() {
                        dofs.push(2);
                    }
                    if stiffness.rx.is_some() {
                        dofs.push(3);
                    }
                    if stiffness.ry.is_some() {
                        dofs.push(4);
                    }
                    if stiffness.rz.is_some() {
                        dofs.push(5);
                    }
                    dofs
                }
            };

            let mut force = Vector3D::zero();
            let mut moment = Vector3D::zero();

            for &local_dof in &constrained_dofs {
                let global_dof = base_dof + local_dof;
                let reaction_value = f_internal[global_dof] - applied_loads[global_dof];

                match local_dof {
                    0 => force.x = reaction_value,
                    1 => force.y = reaction_value,
                    2 => force.z = reaction_value,
                    3 => moment.x = reaction_value,
                    4 => moment.y = reaction_value,
                    5 => moment.z = reaction_value,
                    _ => {}
                }
            }

            reactions.push(SupportReaction::new(
                support.id,
                support.node_id,
                force,
                moment,
            ));
        }

        reactions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::geometry::{Point3D, Vector3D};
    use crate::structure::material::{Material, MaterialType, SteelGrade, SteelStandard};
    use crate::structure::node::{DofMask, Node, NodeId};
    use crate::structure::section::Section;

    #[test]
    fn test_assemble_simple_beam() {
        let model = create_simple_beam_model();
        let k = GlobalAssembler::assemble_global_stiffness(&model);

        assert_eq!(k.nrows(), 12);
        assert_eq!(k.ncols(), 12);
        assert!(k.nnz() > 0);
    }

    fn create_simple_beam_model() -> StructuralModel {
        let node1 = Node {
            id: 0,
            coordinates: Point3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            dof_mask: DofMask::ALL,
        };
        let node2 = Node {
            id: 1,
            coordinates: Point3D {
                x: 10.0,
                y: 0.0,
                z: 0.0,
            },
            dof_mask: DofMask::ALL,
        };

        let material = Material {
            id: 0,
            name: "Steel".to_string(),
            material_type: MaterialType::Steel {
                grade: SteelGrade {
                    standard: SteelStandard::Eurocode3 {
                        grade: "S355".to_string(),
                    },
                    yield_strength: 355e6,
                },
            },
            elastic_modulus: 210e9,
            poisson_ratio: 0.3,
            density: 7850.0,
            thermal_expansion: 12e-6,
            code_reference: None,
        };

        let section = Section {
            area: 0.01,
            inertia_y: 8.333e-6,
            inertia_z: 8.333e-6,
            torsion_constant: 1.0e-6,
        };

        let element = crate::structure::element::Element {
            id: 0,
            material_id: 0,
            element_type: ElementType::Frame3D,
            connectivity: vec![0, 1],
            section,
            local_axes: None,
        };

        use chrono::Utc;
        StructuralModel {
            nodes: vec![node1, node2],
            elements: vec![element],
            materials: vec![material],
            supports: vec![],
            load_cases: vec![],
            metadata: crate::structure::model::ModelMetadata {
                name: "Test".to_string(),
                description: "Test".to_string(),
                created: Utc::now(),
                modified: Utc::now(),
                units: crate::structure::model::UnitSystem::SI,
                coordinate_system: crate::structure::model::CoordinateSystem::Global {
                    x_direction: "East".to_string(),
                    y_direction: "North".to_string(),
                    z_direction: "Up".to_string(),
                },
            },
        }
    }
}
