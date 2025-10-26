use serde::{Deserialize, Serialize};

pub type MaterialId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub id: MaterialId,
    pub name: String,
    pub material_type: MaterialType,
    pub elastic_modulus: f64,
    pub poisson_ratio: f64,
    pub density: f64,
    pub thermal_expansion: f64,
    pub code_reference: Option<CodeReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    Concrete { grade: ConcreteGrade },
    Steel { grade: SteelGrade },
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteGrade {
    pub standard: ConcreteStandard,
    pub characteristic_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConcreteStandard {
    Eurocode2 { grade: String },
    BAEL { grade: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteelGrade {
    pub standard: SteelStandard,
    pub yield_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SteelStandard {
    Eurocode3 { grade: String },
    BAEL { grade: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReference {
    pub standard: String,
    pub section: String,
}

pub mod presets {
    use super::*;

    pub fn concrete_c30_37() -> Material {
        Material {
            id: 0,
            name: "Concrete C30/37".to_string(),
            material_type: MaterialType::Concrete {
                grade: ConcreteGrade {
                    standard: ConcreteStandard::Eurocode2 {
                        grade: "C30/37".to_string(),
                    },
                    characteristic_strength: 30.0,
                },
            },
            elastic_modulus: 33e9,
            poisson_ratio: 0.2,
            density: 2400.0,
            thermal_expansion: 10e-6,
            code_reference: Some(CodeReference {
                standard: "EN 1992-1-1:2004".to_string(),
                section: "Table 3.1".to_string(),
            }),
        }
    }

    pub fn steel_s355() -> Material {
        Material {
            id: 0,
            name: "Steel S355".to_string(),
            material_type: MaterialType::Steel {
                grade: SteelGrade {
                    standard: SteelStandard::Eurocode3 {
                        grade: "S355".to_string(),
                    },
                    yield_strength: 355.0,
                },
            },
            elastic_modulus: 210e9,
            poisson_ratio: 0.3,
            density: 7850.0,
            thermal_expansion: 12e-6,
            code_reference: Some(CodeReference {
                standard: "EN 1993-1-1:2005".to_string(),
                section: "Table 3.1".to_string(),
            }),
        }
    }
}
