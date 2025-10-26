use crate::structure::geometry::Vector3D;
use crate::structure::node::NodeId;
use crate::structure::section::Section;
use serde::{Deserialize, Serialize};

pub type ElementId = usize;
pub type MaterialId = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub id: ElementId,
    pub element_type: ElementType,
    pub connectivity: Vec<NodeId>,
    pub material_id: MaterialId,
    pub section: Section,
    pub local_axes: Option<LocalCoordinateSystem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElementType {
    Beam1D { formulation: BeamFormulation },
    Frame2D { plane: Plane },
    Frame3D,
    Shell2D,
    Solid3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeamFormulation {
    EulerBernoulli,
    Timoshenko,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Plane {
    XY,
    XZ,
    YZ,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalCoordinateSystem {
    pub x_axis: Vector3D,
    pub y_axis: Vector3D,
    pub z_axis: Vector3D,
}

impl LocalCoordinateSystem {
    pub fn from_axes(x_axis: Vector3D, y_axis: Vector3D) -> Self {
        let z_axis = x_axis.cross(&y_axis).normalize();
        Self {
            x_axis: x_axis.normalize(),
            y_axis: y_axis.normalize(),
            z_axis,
        }
    }
}
