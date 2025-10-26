use crate::structure::geometry::Point3D;
use serde::{Deserialize, Serialize};

pub type NodeId = usize;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct DofMask: u8 {
        const TX = 0b00000001;
        const TY = 0b00000010;
        const TZ = 0b00000100;
        const RX = 0b00001000;
        const RY = 0b00010000;
        const RZ = 0b00100000;
        const ALL_TRANSLATIONS = Self::TX.bits() | Self::TY.bits() | Self::TZ.bits();
        const ALL_ROTATIONS = Self::RX.bits() | Self::RY.bits() | Self::RZ.bits();
        const ALL = Self::ALL_TRANSLATIONS.bits() | Self::ALL_ROTATIONS.bits();
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub coordinates: Point3D,
    pub dof_mask: DofMask,
}

impl Node {
    pub fn new(id: NodeId, coordinates: Point3D) -> Self {
        Self {
            id,
            coordinates,
            dof_mask: DofMask::ALL,
        }
    }

    pub fn with_dof_mask(id: NodeId, coordinates: Point3D, dof_mask: DofMask) -> Self {
        Self {
            id,
            coordinates,
            dof_mask,
        }
    }
}
