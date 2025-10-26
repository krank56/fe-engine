use crate::analysis::result::{AnalysisResult, ElementForces, NodalDisplacement, SupportReaction};
use crate::audit::trail::AuditTrail;
use crate::structure::material::CodeReference;
use crate::structure::model::StructuralModel;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportConfig {
    pub include_audit_trail: bool,
    pub include_plots: bool,
    pub language: Language,
    pub code_references: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            include_audit_trail: true,
            include_plots: false,
            language: Language::English,
            code_references: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    French,
    English,
    Spanish,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalculationReport {
    pub title: String,
    pub sections: Vec<ReportSection>,
}

impl CalculationReport {
    pub fn new(title: String) -> Self {
        Self {
            title,
            sections: Vec::new(),
        }
    }

    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportSection {
    Summary {
        max_displacement: f64,
        max_displacement_location: usize,
        max_moment: Option<f64>,
        max_moment_location: Option<usize>,
        max_shear: Option<f64>,
        max_shear_location: Option<usize>,
        total_reaction_force: f64,
        deflection_check: Option<DeflectionCheck>,
    },
    InputData {
        num_nodes: usize,
        num_elements: usize,
        num_supports: usize,
        num_loads: usize,
        materials: Vec<MaterialInfo>,
        span_length: Option<f64>,
    },
    Results {
        displacements: Vec<NodalDisplacement>,
        forces: Vec<ElementForces>,
        reactions: Vec<SupportReaction>,
    },
    AuditTrail {
        entries: Vec<AuditEntry>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialInfo {
    pub name: String,
    pub code_reference: Option<CodeReference>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeflectionCheck {
    pub actual_deflection: f64,
    pub limit_deflection: f64,
    pub span_length: f64,
    pub limit_ratio: String,
    pub passes: bool,
    pub code_reference: CodeReference,
}

impl DeflectionCheck {
    pub fn from_ec2(actual_deflection: f64, span_length: f64) -> Self {
        let limit_deflection = span_length / 250.0;
        Self {
            actual_deflection,
            limit_deflection,
            span_length,
            limit_ratio: "L/250".to_string(),
            passes: actual_deflection <= limit_deflection,
            code_reference: CodeReference {
                standard: "EN 1992-1-1:2004".to_string(),
                section: "§ 7.4.1".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub action: String,
    pub details: Option<String>,
}

impl From<&crate::audit::trail::AuditEntry> for AuditEntry {
    fn from(entry: &crate::audit::trail::AuditEntry) -> Self {
        Self {
            timestamp: entry.timestamp.clone(),
            action: entry.action.clone(),
            details: Some(format!("{:?}", entry.details)),
        }
    }
}

impl AnalysisResult {
    pub fn generate_calculation_report(
        &self,
        config: ReportConfig,
        model: &StructuralModel,
        audit_trail: &AuditTrail,
    ) -> Result<CalculationReport, ReportError> {
        let mut report = CalculationReport::new(format!("Calculation Report - {}", model.metadata.name));

        let max_disp = self.max_displacement();
        let (max_disp_node, _) = self.max_displacement_location();
        
        let (max_moment, max_moment_elem) = self
            .element_forces
            .iter()
            .filter_map(|ef| ef.max_moment().map(|(m, _)| (m, ef.element_id)))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unzip();
        
        let (max_shear, max_shear_elem) = self
            .element_forces
            .iter()
            .filter_map(|ef| ef.max_shear().map(|(s, _)| (s, ef.element_id)))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unzip();
        
        let (total_force, _) = self.total_reactions();
        
        let deflection_check = if config.code_references {
            let span = Self::estimate_span_length(model);
            span.map(|s| DeflectionCheck::from_ec2(max_disp, s))
        } else {
            None
        };
        
        report.add_section(ReportSection::Summary {
            max_displacement: max_disp,
            max_displacement_location: max_disp_node,
            max_moment,
            max_moment_location: max_moment_elem,
            max_shear,
            max_shear_location: max_shear_elem,
            total_reaction_force: total_force.magnitude(),
            deflection_check,
        });

        let materials: Vec<MaterialInfo> = model
            .materials
            .iter()
            .map(|m| MaterialInfo {
                name: m.name.clone(),
                code_reference: m.code_reference.clone(),
            })
            .collect();
        
        let span_length = Self::estimate_span_length(model);

        report.add_section(ReportSection::InputData {
            num_nodes: model.nodes.len(),
            num_elements: model.elements.len(),
            num_supports: model.supports.len(),
            num_loads: model.load_cases.iter().map(|lc| lc.loads.len()).sum(),
            materials,
            span_length,
        });

        report.add_section(ReportSection::Results {
            displacements: self.displacements.clone(),
            forces: self.element_forces.clone(),
            reactions: self.reactions.clone(),
        });

        if config.include_audit_trail {
            let entries: Vec<AuditEntry> = audit_trail.entries.iter().map(AuditEntry::from).collect();
            report.add_section(ReportSection::AuditTrail { entries });
        }

        Ok(report)
    }
    
    pub fn export_markdown_report(
        &self,
        path: impl AsRef<std::path::Path>,
        model: &StructuralModel,
        _load_case: &crate::structure::load::LoadCase,
        language: &str,
    ) -> Result<(), ReportError> {
        let lang = match language {
            "fr" => Language::French,
            "es" => Language::Spanish,
            _ => Language::English,
        };
        
        let config = ReportConfig {
            language: lang,
            code_references: true,
            include_audit_trail: false,
            include_plots: false,
        };
        
        let audit_trail = crate::audit::trail::AuditTrail::new();
        let report = self.generate_calculation_report(config, model, &audit_trail)?;
        report.to_markdown_file(path, lang)?;
        Ok(())
    }
    
    fn estimate_span_length(model: &StructuralModel) -> Option<f64> {
        if model.elements.is_empty() {
            return None;
        }
        
        let mut max_distance: f64 = 0.0;
        for element in &model.elements {
            if element.connectivity.len() >= 2 {
                if let Some(node_i) = model.nodes.iter().find(|n| n.id == element.connectivity[0]) {
                    if let Some(node_j) = model.nodes.iter().find(|n| n.id == element.connectivity[1]) {
                        let dx = node_j.coordinates.x - node_i.coordinates.x;
                        let dy = node_j.coordinates.y - node_i.coordinates.y;
                        let dz = node_j.coordinates.z - node_i.coordinates.z;
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                        max_distance = max_distance.max(distance);
                    }
                }
            }
        }
        
        if max_distance > 0.0 {
            Some(max_distance)
        } else {
            None
        }
    }
}
