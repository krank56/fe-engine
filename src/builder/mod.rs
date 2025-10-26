use chrono::Utc;

use crate::structure::element::{BeamFormulation, Element, ElementId, ElementType, Plane};
use crate::structure::geometry::{Point3D, Vector3D};
use crate::structure::load::{Load, LoadCase, LoadCaseId, LoadDistribution, LoadType};
use crate::structure::material::{Material, MaterialId};
use crate::structure::model::{
    CoordinateSystem, ModelMetadata, StructuralModel, UnitSystem,
};
use crate::structure::node::{DofMask, Node, NodeId};
use crate::structure::section::Section;
use crate::structure::support::{Support, SupportId, SupportType};
use crate::validation::BuildError;

pub struct ModelBuilder {
    name: String,
    description: String,
    units: UnitSystem,
    nodes: Vec<Node>,
    elements: Vec<Element>,
    materials: Vec<Material>,
    supports: Vec<Support>,
    load_cases: Vec<LoadCase>,
    next_node_id: NodeId,
    next_element_id: ElementId,
    next_material_id: MaterialId,
    next_support_id: SupportId,
    next_load_case_id: LoadCaseId,
    current_load_case: Option<LoadCaseBuilder>,
}

impl ModelBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            units: UnitSystem::SI,
            nodes: Vec::new(),
            elements: Vec::new(),
            materials: Vec::new(),
            supports: Vec::new(),
            load_cases: Vec::new(),
            next_node_id: 0,
            next_element_id: 0,
            next_material_id: 0,
            next_support_id: 0,
            next_load_case_id: 0,
            current_load_case: None,
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn units(mut self, units: UnitSystem) -> Self {
        self.units = units;
        self
    }

    pub fn add_node(&mut self, position: Point3D) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(Node {
            id,
            coordinates: position,
            dof_mask: DofMask::ALL,
        });
        id
    }

    pub fn add_material(&mut self, mut material: Material) -> MaterialId {
        let id = self.next_material_id;
        self.next_material_id += 1;
        material.id = id;
        self.materials.push(material);
        id
    }

    pub fn add_beam_element(
        &mut self,
        node_i: NodeId,
        node_j: NodeId,
        material_id: MaterialId,
        section: Section,
    ) -> Result<ElementId, BuildError> {
        if node_i >= self.next_node_id || node_j >= self.next_node_id {
            return Err(BuildError::InvalidNodeReference {
                element_id: self.next_element_id,
                node_id: if node_i >= self.next_node_id {
                    node_i
                } else {
                    node_j
                },
            });
        }

        if material_id >= self.next_material_id {
            return Err(BuildError::InvalidMaterialReference {
                element_id: self.next_element_id,
                material_id,
            });
        }

        let id = self.next_element_id;
        self.next_element_id += 1;

        self.elements.push(Element {
            id,
            element_type: ElementType::Beam1D {
                formulation: BeamFormulation::EulerBernoulli,
            },
            connectivity: vec![node_i, node_j],
            material_id,
            section,
            local_axes: None,
        });

        Ok(id)
    }

    pub fn add_frame_element(
        &mut self,
        node_i: NodeId,
        node_j: NodeId,
        material_id: MaterialId,
        section: Section,
        plane: Plane,
    ) -> Result<ElementId, BuildError> {
        if node_i >= self.next_node_id || node_j >= self.next_node_id {
            return Err(BuildError::InvalidNodeReference {
                element_id: self.next_element_id,
                node_id: if node_i >= self.next_node_id {
                    node_i
                } else {
                    node_j
                },
            });
        }

        if material_id >= self.next_material_id {
            return Err(BuildError::InvalidMaterialReference {
                element_id: self.next_element_id,
                material_id,
            });
        }

        let id = self.next_element_id;
        self.next_element_id += 1;

        self.elements.push(Element {
            id,
            element_type: ElementType::Frame2D { plane },
            connectivity: vec![node_i, node_j],
            material_id,
            section,
            local_axes: None,
        });

        Ok(id)
    }

    pub fn add_support(
        &mut self,
        node_id: NodeId,
        support_type: SupportType,
    ) -> Result<(), BuildError> {
        if node_id >= self.next_node_id {
            return Err(BuildError::InvalidNodeReference {
                element_id: 0,
                node_id,
            });
        }

        let id = self.next_support_id;
        self.next_support_id += 1;

        self.supports.push(Support {
            id,
            node_id,
            support_type,
        });

        Ok(())
    }

    pub fn create_load_case(
        &mut self,
        name: impl Into<String>,
        load_type: LoadType,
    ) -> &mut LoadCaseBuilder {
        if let Some(load_case) = self.current_load_case.take() {
            self.load_cases.push(load_case.into_load_case());
        }

        let id = self.next_load_case_id;
        self.next_load_case_id += 1;

        let element_ids: Vec<ElementId> = self.elements.iter().map(|e| e.id).collect();

        self.current_load_case = Some(LoadCaseBuilder {
            id,
            name: name.into(),
            load_type,
            loads: Vec::new(),
            element_ids,
        });

        self.current_load_case.as_mut().unwrap()
    }

    pub fn build(mut self) -> Result<StructuralModel, BuildError> {
        if let Some(load_case) = self.current_load_case.take() {
            self.load_cases.push(load_case.into_load_case());
        }

        let now = Utc::now();
        let model = StructuralModel {
            nodes: self.nodes,
            elements: self.elements,
            materials: self.materials,
            supports: self.supports,
            load_cases: self.load_cases,
            metadata: ModelMetadata {
                name: self.name,
                description: self.description,
                created: now,
                modified: now,
                units: self.units,
                coordinate_system: CoordinateSystem::Global {
                    x_direction: "East".to_string(),
                    y_direction: "North".to_string(),
                    z_direction: "Up".to_string(),
                },
            },
        };

        model
            .validate()
            .map_err(|errors| BuildError::ValidationFailed { errors })?;

        Ok(model)
    }
}

pub struct LoadCaseBuilder {
    id: LoadCaseId,
    name: String,
    load_type: LoadType,
    loads: Vec<Load>,
    element_ids: Vec<ElementId>,
}

impl LoadCaseBuilder {
    pub fn add_nodal_force(
        &mut self,
        node_id: NodeId,
        force: Vector3D,
        moment: Vector3D,
    ) -> Result<&mut Self, BuildError> {
        self.loads.push(Load::NodalForce {
            node_id,
            force,
            moment,
        });
        Ok(self)
    }

    pub fn add_element_load(
        &mut self,
        element_id: ElementId,
        load_distribution: LoadDistribution,
    ) -> Result<&mut Self, BuildError> {
        self.loads.push(Load::ElementLoad {
            element_id,
            load_distribution,
        });
        Ok(self)
    }

    pub fn add_uniform_load_on_all_elements(
        &mut self,
        intensity: f64,
        direction: Vector3D,
    ) -> Result<&mut Self, BuildError> {
        for &element_id in &self.element_ids {
            self.loads.push(Load::ElementLoad {
                element_id,
                load_distribution: LoadDistribution::UniformDistributed {
                    intensity,
                    direction,
                },
            });
        }
        Ok(self)
    }

    pub fn finish(&mut self) {
    }

    fn into_load_case(self) -> LoadCase {
        LoadCase {
            id: self.id,
            name: self.name,
            load_type: self.load_type,
            loads: self.loads,
        }
    }
}

pub mod materials {
    pub use crate::structure::material::presets::*;
}

pub mod sections {
    pub use crate::structure::section::Section;

    pub fn rectangular(width: f64, height: f64) -> Section {
        Section::rectangular(width, height)
    }

    pub fn circular(diameter: f64) -> Section {
        Section::circular(diameter)
    }
}
