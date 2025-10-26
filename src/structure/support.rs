use serde::{Deserialize, Serialize};

use super::node::NodeId;

pub type SupportId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Support {
    pub id: SupportId,
    pub node_id: NodeId,
    pub support_type: SupportType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportType {
    Fixed,
    Pinned,
    Roller { free_direction: Direction },
    ElasticSpring { stiffness: SpringStiffness },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Direction {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpringStiffness {
    pub tx: Option<f64>,
    pub ty: Option<f64>,
    pub tz: Option<f64>,
    pub rx: Option<f64>,
    pub ry: Option<f64>,
    pub rz: Option<f64>,
}
