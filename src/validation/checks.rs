use super::error::ValidationError;
use crate::structure::element::ElementType;
use crate::structure::model::StructuralModel;
use crate::structure::support::{Direction, SupportType};
use std::collections::{HashMap, HashSet};

pub fn validate_model(model: &StructuralModel) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    errors.extend(validate_non_empty(model));
    errors.extend(validate_unique_ids(model));
    errors.extend(validate_node_references(model));
    errors.extend(validate_geometry(model));
    errors.extend(validate_material_references(model));
    errors.extend(validate_section_properties(model));
    errors.extend(validate_supports(model));
    errors.extend(validate_frame_stability(model));

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_non_empty(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if model.nodes.is_empty() {
        errors.push(ValidationError::EmptyModel {
            message: "No nodes defined".to_string(),
        });
    }

    if model.elements.is_empty() {
        errors.push(ValidationError::EmptyModel {
            message: "No elements defined".to_string(),
        });
    }

    if model.materials.is_empty() {
        errors.push(ValidationError::EmptyModel {
            message: "No materials defined".to_string(),
        });
    }

    errors
}

fn validate_unique_ids(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let mut node_ids = HashSet::new();
    for node in &model.nodes {
        if !node_ids.insert(node.id) {
            errors.push(ValidationError::DuplicateNodeId { node_id: node.id });
        }
    }

    let mut element_ids = HashSet::new();
    for element in &model.elements {
        if !element_ids.insert(element.id) {
            errors.push(ValidationError::DuplicateElementId {
                element_id: element.id,
            });
        }
    }

    errors
}

fn validate_node_references(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let node_ids: HashSet<_> = model.nodes.iter().map(|n| n.id).collect();

    for element in &model.elements {
        for &node_id in &element.connectivity {
            if !node_ids.contains(&node_id) {
                errors.push(ValidationError::InvalidNodeReference {
                    element_id: element.id,
                    node_id,
                });
            }
        }

        let expected_nodes = match &element.element_type {
            ElementType::Beam1D { .. } => 2,
            ElementType::Frame2D { .. } => 2,
            ElementType::Frame3D => 2,
            ElementType::Shell2D => 3,
            ElementType::Solid3D => 8,
        };

        if element.connectivity.len() != expected_nodes {
            errors.push(ValidationError::InvalidElementConnectivity {
                element_id: element.id,
                expected: expected_nodes,
                found: element.connectivity.len(),
            });
        }
    }

    errors
}

fn validate_geometry(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let node_coords: HashMap<_, _> = model.nodes.iter().map(|n| (n.id, &n.coordinates)).collect();

    for element in &model.elements {
        match &element.element_type {
            ElementType::Beam1D { .. } | ElementType::Frame2D { .. } | ElementType::Frame3D => {
                if element.connectivity.len() == 2 {
                    let node1_id = element.connectivity[0];
                    let node2_id = element.connectivity[1];

                    if let (Some(&coord1), Some(&coord2)) =
                        (node_coords.get(&node1_id), node_coords.get(&node2_id))
                    {
                        let length = coord1.distance_to(coord2);
                        if length < 1e-10 {
                            errors.push(ValidationError::ZeroLengthElement {
                                element_id: element.id,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    errors
}

fn validate_material_references(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let material_ids: HashSet<_> = model.materials.iter().map(|m| m.id).collect();

    for element in &model.elements {
        if !material_ids.contains(&element.material_id) {
            errors.push(ValidationError::InvalidMaterialReference {
                element_id: element.id,
                material_id: element.material_id,
            });
        }
    }

    errors
}

fn validate_section_properties(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for element in &model.elements {
        let section = &element.section;

        if section.area <= 0.0 {
            errors.push(ValidationError::InvalidSectionProperties {
                element_id: element.id,
                message: format!("Area must be positive, got {}", section.area),
            });
        }

        if section.inertia_y < 0.0 {
            errors.push(ValidationError::InvalidSectionProperties {
                element_id: element.id,
                message: format!("Inertia Y must be non-negative, got {}", section.inertia_y),
            });
        }

        if section.inertia_z < 0.0 {
            errors.push(ValidationError::InvalidSectionProperties {
                element_id: element.id,
                message: format!("Inertia Z must be non-negative, got {}", section.inertia_z),
            });
        }

        if section.torsion_constant < 0.0 {
            errors.push(ValidationError::InvalidSectionProperties {
                element_id: element.id,
                message: format!(
                    "Torsion constant must be non-negative, got {}",
                    section.torsion_constant
                ),
            });
        }
    }

    for material in &model.materials {
        if material.elastic_modulus <= 0.0 {
            errors.push(ValidationError::InvalidMaterialProperties {
                material_id: material.id,
                message: format!(
                    "Elastic modulus must be positive, got {}",
                    material.elastic_modulus
                ),
            });
        }

        if material.poisson_ratio < -1.0 || material.poisson_ratio > 0.5 {
            errors.push(ValidationError::InvalidMaterialProperties {
                material_id: material.id,
                message: format!(
                    "Poisson ratio must be in range [-1, 0.5], got {}",
                    material.poisson_ratio
                ),
            });
        }

        if material.density < 0.0 {
            errors.push(ValidationError::InvalidMaterialProperties {
                material_id: material.id,
                message: format!("Density must be non-negative, got {}", material.density),
            });
        }
    }

    errors
}

fn validate_supports(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if model.supports.is_empty() {
        errors.push(ValidationError::NoSupportsDefined);
        return errors;
    }

    let node_ids: HashSet<_> = model.nodes.iter().map(|n| n.id).collect();

    for support in &model.supports {
        if !node_ids.contains(&support.node_id) {
            errors.push(ValidationError::InvalidNodeReference {
                element_id: 0,
                node_id: support.node_id,
            });
        }
    }

    errors
}

fn validate_frame_stability(model: &StructuralModel) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if model.supports.is_empty() {
        return errors;
    }

    let has_frame_elements = model.elements.iter().any(|e| {
        matches!(
            e.element_type,
            ElementType::Frame2D { .. } | ElementType::Frame3D
        )
    });

    if !has_frame_elements {
        return errors;
    }

    let mut constrained_dofs = HashSet::new();
    let mut support_nodes = HashSet::new();

    for support in &model.supports {
        support_nodes.insert(support.node_id);

        match &support.support_type {
            SupportType::Fixed => {
                constrained_dofs.insert((support.node_id, "ux"));
                constrained_dofs.insert((support.node_id, "uy"));
                constrained_dofs.insert((support.node_id, "uz"));
                constrained_dofs.insert((support.node_id, "rx"));
                constrained_dofs.insert((support.node_id, "ry"));
                constrained_dofs.insert((support.node_id, "rz"));
            }
            SupportType::Pinned => {
                constrained_dofs.insert((support.node_id, "ux"));
                constrained_dofs.insert((support.node_id, "uy"));
                constrained_dofs.insert((support.node_id, "uz"));
            }
            SupportType::Roller { free_direction } => {
                match free_direction {
                    Direction::X => {
                        constrained_dofs.insert((support.node_id, "uy"));
                        constrained_dofs.insert((support.node_id, "uz"));
                    }
                    Direction::Y => {
                        constrained_dofs.insert((support.node_id, "ux"));
                        constrained_dofs.insert((support.node_id, "uz"));
                    }
                    Direction::Z => {
                        constrained_dofs.insert((support.node_id, "ux"));
                        constrained_dofs.insert((support.node_id, "uy"));
                    }
                }
            }
            SupportType::ElasticSpring { .. } => {}
        }
    }

    let has_2d_frames = model.elements.iter().any(|e| {
        matches!(e.element_type, ElementType::Frame2D { .. })
    });

    if has_2d_frames {
        let ux_constrained = constrained_dofs
            .iter()
            .any(|(_, dof)| *dof == "ux");
        let uy_constrained = constrained_dofs
            .iter()
            .any(|(_, dof)| *dof == "uy");
        let uz_constrained = constrained_dofs
            .iter()
            .any(|(_, dof)| *dof == "uz");
        let rotation_constrained = constrained_dofs
            .iter()
            .any(|(_, dof)| *dof == "rx" || *dof == "ry" || *dof == "rz");

        if !ux_constrained {
            errors.push(ValidationError::InsufficientSupports {
                message: "2D frame requires at least one horizontal restraint to prevent rigid body translation".to_string(),
            });
        }
        if !uy_constrained {
            errors.push(ValidationError::InsufficientSupports {
                message: "2D frame requires at least one vertical restraint to prevent rigid body translation".to_string(),
            });
        }
        if !uz_constrained {
            errors.push(ValidationError::InsufficientSupports {
                message: "2D frame requires at least one out-of-plane restraint to prevent rigid body translation".to_string(),
            });
        }

        let multiple_supports = support_nodes.len() >= 2;
        if !multiple_supports && !rotation_constrained {
            errors.push(ValidationError::InsufficientSupports {
                message: "2D frame requires multiple supports or rotational restraint to prevent rigid body rotation".to_string(),
            });
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{
        element::{Element, ElementType},
        geometry::Point3D,
        material::Material,
        node::Node,
        section::Section,
        support::{Direction, Support, SupportType},
    };
    use chrono::Utc;

    #[test]
    fn test_validate_frame_stability_with_fixed_supports() {
        let mut model = create_simple_frame();
        model.supports = vec![
            Support {
                id: 1,
                node_id: 1,
                support_type: SupportType::Fixed,
            },
            Support {
                id: 2,
                node_id: 2,
                support_type: SupportType::Fixed,
            },
        ];

        let errors = validate_frame_stability(&model);
        assert_eq!(errors.len(), 0, "Fixed supports should satisfy stability");
    }

    #[test]
    fn test_validate_frame_stability_with_pinned_supports() {
        let mut model = create_simple_frame();
        model.supports = vec![
            Support {
                id: 1,
                node_id: 1,
                support_type: SupportType::Pinned,
            },
            Support {
                id: 2,
                node_id: 2,
                support_type: SupportType::Pinned,
            },
        ];

        let errors = validate_frame_stability(&model);
        assert_eq!(errors.len(), 0, "Two pinned supports should satisfy stability");
    }

    #[test]
    fn test_validate_frame_stability_insufficient_horizontal() {
        let mut model = create_simple_frame();
        model.supports = vec![Support {
            id: 1,
            node_id: 1,
            support_type: SupportType::Roller {
                free_direction: Direction::X,
            },
        }];

        let errors = validate_frame_stability(&model);
        assert!(
            errors.len() > 0,
            "Should detect missing horizontal restraint"
        );
        assert!(errors.iter().any(|e| matches!(e,
            ValidationError::InsufficientSupports { message } if message.contains("horizontal")
        )));
    }

    #[test]
    fn test_validate_frame_stability_insufficient_vertical() {
        let mut model = create_simple_frame();
        model.supports = vec![Support {
            id: 1,
            node_id: 1,
            support_type: SupportType::Roller {
                free_direction: Direction::Y,
            },
        }];

        let errors = validate_frame_stability(&model);
        assert!(errors.len() > 0, "Should detect missing vertical restraint");
        assert!(errors.iter().any(|e| matches!(e,
            ValidationError::InsufficientSupports { message } if message.contains("vertical")
        )));
    }

    #[test]
    fn test_validate_frame_stability_insufficient_rotation() {
        let mut model = create_simple_frame();
        model.supports = vec![Support {
            id: 1,
            node_id: 1,
            support_type: SupportType::Pinned,
        }];

        let errors = validate_frame_stability(&model);
        assert!(
            errors.len() > 0,
            "Should detect potential rotation instability"
        );
        assert!(errors.iter().any(|e| matches!(e,
            ValidationError::InsufficientSupports { message } if message.contains("rotation")
        )));
    }

    #[test]
    fn test_validate_frame_stability_non_frame_elements() {
        let mut model = create_simple_frame();
        model.elements[0].element_type = ElementType::Beam1D {
            formulation: crate::structure::element::BeamFormulation::EulerBernoulli,
        };
        model.supports = vec![Support {
            id: 1,
            node_id: 1,
            support_type: SupportType::Pinned,
        }];

        let errors = validate_frame_stability(&model);
        assert_eq!(
            errors.len(),
            0,
            "Should skip validation for non-frame elements"
        );
    }

    fn create_simple_frame() -> StructuralModel {
        use crate::structure::node::DofMask;
        use crate::structure::material::{ConcreteGrade, ConcreteStandard};

        StructuralModel {
            nodes: vec![
                Node {
                    id: 1,
                    coordinates: Point3D {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    dof_mask: DofMask::ALL,
                },
                Node {
                    id: 2,
                    coordinates: Point3D {
                        x: 5.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    dof_mask: DofMask::ALL,
                },
            ],
            elements: vec![Element {
                id: 1,
                connectivity: vec![1, 2],
                element_type: ElementType::Frame2D {
                    plane: crate::structure::element::Plane::XZ,
                },
                material_id: 1,
                section: Section {
                    area: 0.09,
                    inertia_y: 0.000675,
                    inertia_z: 0.000675,
                    torsion_constant: 0.001,
                },
                local_axes: None,
            }],
            materials: vec![Material {
                id: 1,
                name: "Concrete".to_string(),
                elastic_modulus: 30e9,
                poisson_ratio: 0.2,
                density: 2400.0,
                thermal_expansion: 1e-5,
                code_reference: None,
                material_type: crate::structure::material::MaterialType::Concrete {
                    grade: ConcreteGrade {
                        standard: ConcreteStandard::Eurocode2 {
                            grade: "C30/37".to_string(),
                        },
                        characteristic_strength: 30e6,
                    },
                },
            }],
            supports: vec![],
            load_cases: vec![],
            metadata: crate::structure::model::ModelMetadata {
                name: "Test Frame".to_string(),
                description: "Test".to_string(),
                created: Utc::now(),
                modified: Utc::now(),
                units: crate::structure::model::UnitSystem::SI,
                coordinate_system: crate::structure::model::CoordinateSystem::Global {
                    x_direction: "Right".to_string(),
                    y_direction: "Up".to_string(),
                    z_direction: "Out".to_string(),
                },
            },
        }
    }
}
