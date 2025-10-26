use crate::export::report::{CalculationReport, Language, ReportSection};
use std::io::Write;
use std::path::Path;

impl CalculationReport {
    pub fn to_markdown(&self, language: Language) -> String {
        let mut output = String::new();

        match language {
            Language::French => {
                output.push_str(&format!("# Note de Calcul - {}\n\n", self.title));
                output.push_str(&format!("**Date**: {}\n", chrono::Utc::now().format("%Y-%m-%d")));
                output.push_str("**Ingénieur**: [À compléter]\n");
                output.push_str(&format!("**Logiciel**: Structure AI FE Engine v{}\n\n", env!("CARGO_PKG_VERSION")));
            }
            Language::English => {
                output.push_str(&format!("# Calculation Report - {}\n\n", self.title));
                output.push_str(&format!("**Date**: {}\n", chrono::Utc::now().format("%Y-%m-%d")));
                output.push_str("**Engineer**: [To be completed]\n");
                output.push_str(&format!("**Software**: Structure AI FE Engine v{}\n\n", env!("CARGO_PKG_VERSION")));
            }
            Language::Spanish => {
                output.push_str(&format!("# Nota de Cálculo - {}\n\n", self.title));
                output.push_str(&format!("**Fecha**: {}\n", chrono::Utc::now().format("%Y-%m-%d")));
                output.push_str("**Ingeniero**: [Por completar]\n");
                output.push_str(&format!("**Software**: Structure AI FE Engine v{}\n\n", env!("CARGO_PKG_VERSION")));
            }
        }

        for (idx, section) in self.sections.iter().enumerate() {
            output.push_str(&render_section(section, idx + 1, language));
            output.push_str("\n");
        }

        output
    }

    pub fn to_markdown_file(&self, path: impl AsRef<Path>, language: Language) -> std::io::Result<()> {
        let markdown = self.to_markdown(language);
        let mut file = std::fs::File::create(path)?;
        file.write_all(markdown.as_bytes())?;
        Ok(())
    }
}

fn render_section(section: &ReportSection, section_num: usize, language: Language) -> String {
    let mut output = String::new();

    match section {
        ReportSection::Summary {
            max_displacement,
            max_displacement_location,
            max_moment,
            max_moment_location,
            max_shear,
            max_shear_location,
            total_reaction_force,
            deflection_check,
        } => {
            match language {
                Language::French => {
                    output.push_str(&format!("## {}. Résumé\n\n", section_num));
                    output.push_str(&format!("- Flèche maximale: **{:.3} mm** (nœud {})\n", max_displacement * 1000.0, max_displacement_location));
                    if let Some(check) = deflection_check {
                        output.push_str(&format!(
                            "  - Limite admissible ({}, {} {}): {:.3} mm {}\n",
                            check.limit_ratio,
                            check.code_reference.standard,
                            check.code_reference.section,
                            check.limit_deflection * 1000.0,
                            if check.passes { "✓" } else { "✗" }
                        ));
                    }
                    if let Some(moment) = max_moment {
                        output.push_str(&format!("- Moment maximal: **{:.1} kNm**", moment / 1000.0));
                        if let Some(elem) = max_moment_location {
                            output.push_str(&format!(" (élément {})", elem));
                        }
                        output.push_str("\n");
                    }
                    if let Some(shear) = max_shear {
                        output.push_str(&format!("- Effort tranchant maximal: **{:.1} kN**", shear / 1000.0));
                        if let Some(elem) = max_shear_location {
                            output.push_str(&format!(" (élément {})", elem));
                        }
                        output.push_str("\n");
                    }
                    output.push_str(&format!("- Réaction totale: **{:.1} kN**\n", total_reaction_force / 1000.0));
                }
                Language::English => {
                    output.push_str(&format!("## {}. Summary\n\n", section_num));
                    output.push_str(&format!("- Maximum displacement: **{:.3} mm** (node {})\n", max_displacement * 1000.0, max_displacement_location));
                    if let Some(check) = deflection_check {
                        output.push_str(&format!(
                            "  - Allowable limit ({}, {} {}): {:.3} mm {}\n",
                            check.limit_ratio,
                            check.code_reference.standard,
                            check.code_reference.section,
                            check.limit_deflection * 1000.0,
                            if check.passes { "✓" } else { "✗" }
                        ));
                    }
                    if let Some(moment) = max_moment {
                        output.push_str(&format!("- Maximum moment: **{:.1} kNm**", moment / 1000.0));
                        if let Some(elem) = max_moment_location {
                            output.push_str(&format!(" (element {})", elem));
                        }
                        output.push_str("\n");
                    }
                    if let Some(shear) = max_shear {
                        output.push_str(&format!("- Maximum shear: **{:.1} kN**", shear / 1000.0));
                        if let Some(elem) = max_shear_location {
                            output.push_str(&format!(" (element {})", elem));
                        }
                        output.push_str("\n");
                    }
                    output.push_str(&format!("- Total reaction: **{:.1} kN**\n", total_reaction_force / 1000.0));
                }
                Language::Spanish => {
                    output.push_str(&format!("## {}. Resumen\n\n", section_num));
                    output.push_str(&format!("- Flecha máxima: **{:.3} mm** (nodo {})\n", max_displacement * 1000.0, max_displacement_location));
                    if let Some(check) = deflection_check {
                        output.push_str(&format!(
                            "  - Límite admisible ({}, {} {}): {:.3} mm {}\n",
                            check.limit_ratio,
                            check.code_reference.standard,
                            check.code_reference.section,
                            check.limit_deflection * 1000.0,
                            if check.passes { "✓" } else { "✗" }
                        ));
                    }
                    if let Some(moment) = max_moment {
                        output.push_str(&format!("- Momento máximo: **{:.1} kNm**", moment / 1000.0));
                        if let Some(elem) = max_moment_location {
                            output.push_str(&format!(" (elemento {})", elem));
                        }
                        output.push_str("\n");
                    }
                    if let Some(shear) = max_shear {
                        output.push_str(&format!("- Cortante máximo: **{:.1} kN**", shear / 1000.0));
                        if let Some(elem) = max_shear_location {
                            output.push_str(&format!(" (elemento {})", elem));
                        }
                        output.push_str("\n");
                    }
                    output.push_str(&format!("- Reacción total: **{:.1} kN**\n", total_reaction_force / 1000.0));
                }
            }
        }
        ReportSection::InputData {
            num_nodes,
            num_elements,
            num_supports,
            num_loads,
            materials,
            span_length,
        } => {
            match language {
                Language::French => {
                    output.push_str(&format!("## {}. Données d'Entrée\n\n", section_num));
                    output.push_str(&format!("- Nombre de nœuds: {}\n", num_nodes));
                    output.push_str(&format!("- Nombre d'éléments: {}\n", num_elements));
                    output.push_str(&format!("- Nombre d'appuis: {}\n", num_supports));
                    output.push_str(&format!("- Nombre de charges: {}\n", num_loads));
                    if let Some(span) = span_length {
                        output.push_str(&format!("- Portée estimée: {:.2} m\n", span));
                    }
                    if !materials.is_empty() {
                        output.push_str("\n### Matériaux\n\n");
                        for mat in materials {
                            if let Some(ref code_ref) = mat.code_reference {
                                output.push_str(&format!("- **{}** ({}, {})\n", mat.name, code_ref.standard, code_ref.section));
                            } else {
                                output.push_str(&format!("- **{}**\n", mat.name));
                            }
                        }
                    }
                }
                Language::English => {
                    output.push_str(&format!("## {}. Input Data\n\n", section_num));
                    output.push_str(&format!("- Number of nodes: {}\n", num_nodes));
                    output.push_str(&format!("- Number of elements: {}\n", num_elements));
                    output.push_str(&format!("- Number of supports: {}\n", num_supports));
                    output.push_str(&format!("- Number of loads: {}\n", num_loads));
                    if let Some(span) = span_length {
                        output.push_str(&format!("- Estimated span: {:.2} m\n", span));
                    }
                    if !materials.is_empty() {
                        output.push_str("\n### Materials\n\n");
                        for mat in materials {
                            if let Some(ref code_ref) = mat.code_reference {
                                output.push_str(&format!("- **{}** ({}, {})\n", mat.name, code_ref.standard, code_ref.section));
                            } else {
                                output.push_str(&format!("- **{}**\n", mat.name));
                            }
                        }
                    }
                }
                Language::Spanish => {
                    output.push_str(&format!("## {}. Datos de Entrada\n\n", section_num));
                    output.push_str(&format!("- Número de nodos: {}\n", num_nodes));
                    output.push_str(&format!("- Número de elementos: {}\n", num_elements));
                    output.push_str(&format!("- Número de apoyos: {}\n", num_supports));
                    output.push_str(&format!("- Número de cargas: {}\n", num_loads));
                    if let Some(span) = span_length {
                        output.push_str(&format!("- Luz estimada: {:.2} m\n", span));
                    }
                    if !materials.is_empty() {
                        output.push_str("\n### Materiales\n\n");
                        for mat in materials {
                            if let Some(ref code_ref) = mat.code_reference {
                                output.push_str(&format!("- **{}** ({}, {})\n", mat.name, code_ref.standard, code_ref.section));
                            } else {
                                output.push_str(&format!("- **{}**\n", mat.name));
                            }
                        }
                    }
                }
            }
        }
        ReportSection::Results {
            displacements,
            forces: _,
            reactions,
        } => {
            match language {
                Language::French => {
                    output.push_str(&format!("## {}. Résultats\n\n", section_num));
                    output.push_str(&format!("### {}.1 Déplacements\n\n", section_num));
                    output.push_str("| Nœud | TX (mm) | TY (mm) | TZ (mm) | RX (rad) | RY (rad) | RZ (rad) |\n");
                    output.push_str("|------|---------|---------|---------|----------|----------|----------|\n");
                    for disp in displacements.iter().take(10) {
                        output.push_str(&format!(
                            "| {} | {:.3} | {:.3} | {:.3} | {:.6} | {:.6} | {:.6} |\n",
                            disp.node_id,
                            disp.translation.x * 1000.0,
                            disp.translation.y * 1000.0,
                            disp.translation.z * 1000.0,
                            disp.rotation.x,
                            disp.rotation.y,
                            disp.rotation.z
                        ));
                    }
                    if displacements.len() > 10 {
                        output.push_str(&format!("\n*({} nœuds supplémentaires omis)*\n", displacements.len() - 10));
                    }

                    output.push_str(&format!("\n### {}.2 Réactions d'Appui\n\n", section_num));
                    output.push_str("| Appui | Nœud | FX (kN) | FY (kN) | FZ (kN) | MX (kNm) | MY (kNm) | MZ (kNm) |\n");
                    output.push_str("|-------|------|---------|---------|---------|----------|----------|----------|\n");
                    for reaction in reactions {
                        output.push_str(&format!(
                            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                            reaction.support_id,
                            reaction.node_id,
                            reaction.force.x / 1000.0,
                            reaction.force.y / 1000.0,
                            reaction.force.z / 1000.0,
                            reaction.moment.x / 1000.0,
                            reaction.moment.y / 1000.0,
                            reaction.moment.z / 1000.0
                        ));
                    }
                }
                Language::English => {
                    output.push_str(&format!("## {}. Results\n\n", section_num));
                    output.push_str(&format!("### {}.1 Displacements\n\n", section_num));
                    output.push_str("| Node | TX (mm) | TY (mm) | TZ (mm) | RX (rad) | RY (rad) | RZ (rad) |\n");
                    output.push_str("|------|---------|---------|---------|----------|----------|----------|\n");
                    for disp in displacements.iter().take(10) {
                        output.push_str(&format!(
                            "| {} | {:.3} | {:.3} | {:.3} | {:.6} | {:.6} | {:.6} |\n",
                            disp.node_id,
                            disp.translation.x * 1000.0,
                            disp.translation.y * 1000.0,
                            disp.translation.z * 1000.0,
                            disp.rotation.x,
                            disp.rotation.y,
                            disp.rotation.z
                        ));
                    }
                    if displacements.len() > 10 {
                        output.push_str(&format!("\n*({} additional nodes omitted)*\n", displacements.len() - 10));
                    }

                    output.push_str(&format!("\n### {}.2 Support Reactions\n\n", section_num));
                    output.push_str("| Support | Node | FX (kN) | FY (kN) | FZ (kN) | MX (kNm) | MY (kNm) | MZ (kNm) |\n");
                    output.push_str("|---------|------|---------|---------|---------|----------|----------|----------|\n");
                    for reaction in reactions {
                        output.push_str(&format!(
                            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                            reaction.support_id,
                            reaction.node_id,
                            reaction.force.x / 1000.0,
                            reaction.force.y / 1000.0,
                            reaction.force.z / 1000.0,
                            reaction.moment.x / 1000.0,
                            reaction.moment.y / 1000.0,
                            reaction.moment.z / 1000.0
                        ));
                    }
                }
                Language::Spanish => {
                    output.push_str(&format!("## {}. Resultados\n\n", section_num));
                    output.push_str(&format!("### {}.1 Desplazamientos\n\n", section_num));
                    output.push_str("| Nodo | TX (mm) | TY (mm) | TZ (mm) | RX (rad) | RY (rad) | RZ (rad) |\n");
                    output.push_str("|------|---------|---------|---------|----------|----------|----------|\n");
                    for disp in displacements.iter().take(10) {
                        output.push_str(&format!(
                            "| {} | {:.3} | {:.3} | {:.3} | {:.6} | {:.6} | {:.6} |\n",
                            disp.node_id,
                            disp.translation.x * 1000.0,
                            disp.translation.y * 1000.0,
                            disp.translation.z * 1000.0,
                            disp.rotation.x,
                            disp.rotation.y,
                            disp.rotation.z
                        ));
                    }
                    if displacements.len() > 10 {
                        output.push_str(&format!("\n*({} nodos adicionales omitidos)*\n", displacements.len() - 10));
                    }

                    output.push_str(&format!("\n### {}.2 Reacciones de Apoyo\n\n", section_num));
                    output.push_str("| Apoyo | Nodo | FX (kN) | FY (kN) | FZ (kN) | MX (kNm) | MY (kNm) | MZ (kNm) |\n");
                    output.push_str("|-------|------|---------|---------|---------|----------|----------|----------|\n");
                    for reaction in reactions {
                        output.push_str(&format!(
                            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                            reaction.support_id,
                            reaction.node_id,
                            reaction.force.x / 1000.0,
                            reaction.force.y / 1000.0,
                            reaction.force.z / 1000.0,
                            reaction.moment.x / 1000.0,
                            reaction.moment.y / 1000.0,
                            reaction.moment.z / 1000.0
                        ));
                    }
                }
            }
        }
        ReportSection::AuditTrail { entries } => {
            match language {
                Language::French => {
                    output.push_str(&format!("## {}. Trace d'Audit\n\n", section_num));
                    output.push_str("| Horodatage | Action |\n");
                    output.push_str("|------------|--------|\n");
                    for entry in entries.iter().take(20) {
                        output.push_str(&format!("| {} | {} |\n", entry.timestamp, entry.action));
                    }
                    if entries.len() > 20 {
                        output.push_str(&format!("\n*({} entrées supplémentaires omises)*\n", entries.len() - 20));
                    }
                }
                Language::English => {
                    output.push_str(&format!("## {}. Audit Trail\n\n", section_num));
                    output.push_str("| Timestamp | Action |\n");
                    output.push_str("|-----------|--------|\n");
                    for entry in entries.iter().take(20) {
                        output.push_str(&format!("| {} | {} |\n", entry.timestamp, entry.action));
                    }
                    if entries.len() > 20 {
                        output.push_str(&format!("\n*({} additional entries omitted)*\n", entries.len() - 20));
                    }
                }
                Language::Spanish => {
                    output.push_str(&format!("## {}. Registro de Auditoría\n\n", section_num));
                    output.push_str("| Marca de Tiempo | Acción |\n");
                    output.push_str("|-----------------|--------|\n");
                    for entry in entries.iter().take(20) {
                        output.push_str(&format!("| {} | {} |\n", entry.timestamp, entry.action));
                    }
                    if entries.len() > 20 {
                        output.push_str(&format!("\n*({} entradas adicionales omitidas)*\n", entries.len() - 20));
                    }
                }
            }
        }
    }

    output
}
