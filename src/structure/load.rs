use serde::{Deserialize, Serialize};

use super::element::ElementId;
use super::geometry::Vector3D;
use super::node::NodeId;

pub type LoadCaseId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCase {
    pub id: LoadCaseId,
    pub name: String,
    pub load_type: LoadType,
    pub loads: Vec<Load>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadType {
    Dead,
    Live,
    Wind,
    Seismic,
    Temperature,
    Settlement,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Load {
    NodalForce {
        node_id: NodeId,
        force: Vector3D,
        moment: Vector3D,
    },
    ElementLoad {
        element_id: ElementId,
        load_distribution: LoadDistribution,
    },
    ImposedDisplacement {
        node_id: NodeId,
        displacement: Vector3D,
        rotation: Vector3D,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadDistribution {
    UniformDistributed {
        intensity: f64,
        direction: Vector3D,
    },
    LinearlyVarying {
        start_intensity: f64,
        end_intensity: f64,
        direction: Vector3D,
    },
    PointLoad {
        position: f64,
        force: Vector3D,
        moment: Vector3D,
    },
}
