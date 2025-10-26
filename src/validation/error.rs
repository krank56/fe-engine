use crate::structure::element::ElementId;
use crate::structure::node::NodeId;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    InvalidNodeReference {
        element_id: ElementId,
        node_id: NodeId,
    },
    DuplicateNodeId {
        node_id: NodeId,
    },
    DuplicateElementId {
        element_id: ElementId,
    },
    ZeroLengthElement {
        element_id: ElementId,
    },
    InvalidElementConnectivity {
        element_id: ElementId,
        expected: usize,
        found: usize,
    },
    InvalidMaterialReference {
        element_id: ElementId,
        material_id: usize,
    },
    NoSupportsDefined,
    InsufficientSupports {
        message: String,
    },
    ConflictingSupports {
        node_id: NodeId,
        message: String,
    },
    InvalidSectionProperties {
        element_id: ElementId,
        message: String,
    },
    InvalidMaterialProperties {
        material_id: usize,
        message: String,
    },
    EmptyModel {
        message: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidNodeReference {
                element_id,
                node_id,
            } => {
                write!(
                    f,
                    "Element {} references non-existent node {}",
                    element_id, node_id
                )
            }
            ValidationError::DuplicateNodeId { node_id } => {
                write!(f, "Duplicate node ID: {}", node_id)
            }
            ValidationError::DuplicateElementId { element_id } => {
                write!(f, "Duplicate element ID: {}", element_id)
            }
            ValidationError::ZeroLengthElement { element_id } => {
                write!(f, "Element {} has zero length", element_id)
            }
            ValidationError::InvalidElementConnectivity {
                element_id,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Element {} has invalid connectivity: expected {} nodes, found {}",
                    element_id, expected, found
                )
            }
            ValidationError::InvalidMaterialReference {
                element_id,
                material_id,
            } => {
                write!(
                    f,
                    "Element {} references non-existent material {}",
                    element_id, material_id
                )
            }
            ValidationError::NoSupportsDefined => {
                write!(f, "Model has no supports defined (unstable structure)")
            }
            ValidationError::InsufficientSupports { message } => {
                write!(f, "Insufficient supports: {}", message)
            }
            ValidationError::ConflictingSupports { node_id, message } => {
                write!(f, "Conflicting supports at node {}: {}", node_id, message)
            }
            ValidationError::InvalidSectionProperties {
                element_id,
                message,
            } => {
                write!(
                    f,
                    "Invalid section properties for element {}: {}",
                    element_id, message
                )
            }
            ValidationError::InvalidMaterialProperties {
                material_id,
                message,
            } => {
                write!(
                    f,
                    "Invalid material properties for material {}: {}",
                    material_id, message
                )
            }
            ValidationError::EmptyModel { message } => {
                write!(f, "Empty model: {}", message)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    DuplicateId {
        message: String,
    },
    InvalidProperty {
        message: String,
    },
    MissingRequiredField {
        field: String,
    },
    InconsistentState {
        message: String,
    },
    InvalidNodeReference {
        element_id: ElementId,
        node_id: NodeId,
    },
    InvalidMaterialReference {
        element_id: ElementId,
        material_id: usize,
    },
    ValidationFailed {
        errors: Vec<ValidationError>,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::DuplicateId { message } => {
                write!(f, "Duplicate ID: {}", message)
            }
            BuildError::InvalidProperty { message } => {
                write!(f, "Invalid property: {}", message)
            }
            BuildError::MissingRequiredField { field } => {
                write!(f, "Missing required field: {}", field)
            }
            BuildError::InconsistentState { message } => {
                write!(f, "Inconsistent state: {}", message)
            }
            BuildError::InvalidNodeReference {
                element_id,
                node_id,
            } => {
                write!(
                    f,
                    "Element {} references non-existent node {}",
                    element_id, node_id
                )
            }
            BuildError::InvalidMaterialReference {
                element_id,
                material_id,
            } => {
                write!(
                    f,
                    "Element {} references non-existent material {}",
                    element_id, material_id
                )
            }
            BuildError::ValidationFailed { errors } => {
                write!(f, "Model validation failed with {} errors", errors.len())
            }
        }
    }
}

impl std::error::Error for BuildError {}
