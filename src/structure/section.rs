use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub area: f64,
    pub inertia_y: f64,
    pub inertia_z: f64,
    pub torsion_constant: f64,
}

impl Section {
    pub fn new(area: f64, inertia_y: f64, inertia_z: f64, torsion_constant: f64) -> Self {
        Self {
            area,
            inertia_y,
            inertia_z,
            torsion_constant,
        }
    }

    pub fn rectangular(width: f64, height: f64) -> Self {
        let area = width * height;
        let iy = (width * height.powi(3)) / 12.0;
        let iz = (height * width.powi(3)) / 12.0;
        let j = area.powi(2) / (40.0 * (width + height));

        Self {
            area,
            inertia_y: iy,
            inertia_z: iz,
            torsion_constant: j,
        }
    }

    pub fn circular(diameter: f64) -> Self {
        let radius = diameter / 2.0;
        let area = std::f64::consts::PI * radius.powi(2);
        let i = (std::f64::consts::PI * radius.powi(4)) / 4.0;
        let j = (std::f64::consts::PI * radius.powi(4)) / 2.0;

        Self {
            area,
            inertia_y: i,
            inertia_z: i,
            torsion_constant: j,
        }
    }
}
