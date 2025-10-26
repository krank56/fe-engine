use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::element::Element;
use super::load::LoadCase;
use super::material::Material;
use super::node::Node;
use super::support::Support;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralModel {
    pub nodes: Vec<Node>,
    pub elements: Vec<Element>,
    pub materials: Vec<Material>,
    pub supports: Vec<Support>,
    pub load_cases: Vec<LoadCase>,
    pub metadata: ModelMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: String,
    pub created: DateTime<Utc>,
    pub modified: DateTime<Utc>,
    pub units: UnitSystem,
    pub coordinate_system: CoordinateSystem,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UnitSystem {
    SI,
    Imperial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinateSystem {
    Global {
        x_direction: String,
        y_direction: String,
        z_direction: String,
    },
}

impl StructuralModel {
    pub fn validate(&self) -> Result<(), Vec<crate::validation::ValidationError>> {
        crate::validation::validate_model(self)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
