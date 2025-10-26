use crate::analysis::result::AnalysisResult;
use crate::structure::element::ElementId;
use std::io::Write;
use std::path::Path;

impl AnalysisResult {
    pub fn export_displacements_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        
        writeln!(file, "NodeID,TX(m),TY(m),TZ(m),RX(rad),RY(rad),RZ(rad)")?;
        
        for disp in &self.displacements {
            writeln!(
                file,
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                disp.node_id,
                disp.translation.x,
                disp.translation.y,
                disp.translation.z,
                disp.rotation.x,
                disp.rotation.y,
                disp.rotation.z
            )?;
        }
        
        Ok(())
    }

    pub fn export_reactions_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        
        writeln!(file, "SupportID,NodeID,FX(N),FY(N),FZ(N),MX(Nm),MY(Nm),MZ(Nm)")?;
        
        for reaction in &self.reactions {
            writeln!(
                file,
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                reaction.support_id,
                reaction.node_id,
                reaction.force.x,
                reaction.force.y,
                reaction.force.z,
                reaction.moment.x,
                reaction.moment.y,
                reaction.moment.z
            )?;
        }
        
        Ok(())
    }

    pub fn export_beam_forces_csv(
        &self,
        element_id: ElementId,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        let element_forces = self
            .element_forces
            .iter()
            .find(|ef| ef.element_id == element_id)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Element {} not found in results", element_id),
                )
            })?;

        let mut file = std::fs::File::create(path)?;
        
        writeln!(
            file,
            "Position(m),Axial(N),ShearY(N),ShearZ(N),MomentY(Nm),MomentZ(Nm),Torsion(Nm)"
        )?;

        match &element_forces.forces {
            crate::analysis::result::ElementForceComponents::Beam {
                axial,
                shear_y,
                shear_z,
                moment_y,
                moment_z,
                torsion,
                evaluation_points,
            } => {
                for i in 0..evaluation_points.len() {
                    writeln!(
                        file,
                        "{:.6},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                        evaluation_points[i],
                        axial[i],
                        shear_y[i],
                        shear_z[i],
                        moment_y[i],
                        moment_z[i],
                        torsion[i]
                    )?;
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Element is not a beam",
                ));
            }
        }

        Ok(())
    }
}
