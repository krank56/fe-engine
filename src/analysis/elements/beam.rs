use nalgebra::{Matrix6, SMatrix, Vector3};

use crate::structure::geometry::{Point3D, Vector3D};

pub struct BeamElement {
    pub length: f64,
    pub area: f64,
    pub elastic_modulus: f64,
    pub shear_modulus: f64,
    pub inertia_y: f64,
    pub inertia_z: f64,
    pub torsion_constant: f64,
}

impl BeamElement {
    pub fn new(
        node_i: &Point3D,
        node_j: &Point3D,
        elastic_modulus: f64,
        shear_modulus: f64,
        area: f64,
        inertia_y: f64,
        inertia_z: f64,
        torsion_constant: f64,
    ) -> Self {
        let dx = node_j.x - node_i.x;
        let dy = node_j.y - node_i.y;
        let dz = node_j.z - node_i.z;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();

        Self {
            length,
            area,
            elastic_modulus,
            shear_modulus,
            inertia_y,
            inertia_z,
            torsion_constant,
        }
    }

    pub fn local_stiffness_matrix(&self) -> SMatrix<f64, 12, 12> {
        let e = self.elastic_modulus;
        let g = self.shear_modulus;
        let a = self.area;
        let l = self.length;
        let iy = self.inertia_y;
        let iz = self.inertia_z;
        let j = self.torsion_constant;

        let ea_l = (e * a) / l;
        let gj_l = (g * j) / l;
        let eiy_l3 = (e * iy) / (l * l * l);
        let eiz_l3 = (e * iz) / (l * l * l);

        let k12_y = 12.0 * eiy_l3;
        let k12_z = 12.0 * eiz_l3;

        let k6_y = 6.0 * eiy_l3 * l;
        let k6_z = 6.0 * eiz_l3 * l;

        let k4_y = 4.0 * eiy_l3 * l * l;
        let k4_z = 4.0 * eiz_l3 * l * l;

        let k2_y = 2.0 * eiy_l3 * l * l;
        let k2_z = 2.0 * eiz_l3 * l * l;

        #[rustfmt::skip]
        let k_local = SMatrix::<f64, 12, 12>::from_row_slice(&[
            ea_l,     0.0,      0.0,      0.0,      0.0,      0.0,     -ea_l,     0.0,      0.0,      0.0,      0.0,      0.0,
            0.0,      k12_z,    0.0,      0.0,      0.0,      k6_z,     0.0,     -k12_z,    0.0,      0.0,      0.0,      k6_z,
            0.0,      0.0,      k12_y,    0.0,      k6_y,     0.0,      0.0,      0.0,     -k12_y,    0.0,      k6_y,     0.0,
            0.0,      0.0,      0.0,      gj_l,     0.0,      0.0,      0.0,      0.0,      0.0,     -gj_l,     0.0,      0.0,
            0.0,      0.0,      k6_y,     0.0,      k4_y,     0.0,      0.0,      0.0,     -k6_y,     0.0,      k2_y,     0.0,
            0.0,      k6_z,     0.0,      0.0,      0.0,      k4_z,     0.0,     -k6_z,     0.0,      0.0,      0.0,      k2_z,
           -ea_l,     0.0,      0.0,      0.0,      0.0,      0.0,      ea_l,     0.0,      0.0,      0.0,      0.0,      0.0,
            0.0,     -k12_z,    0.0,      0.0,      0.0,     -k6_z,     0.0,      k12_z,    0.0,      0.0,      0.0,     -k6_z,
            0.0,      0.0,     -k12_y,    0.0,     -k6_y,     0.0,      0.0,      0.0,      k12_y,    0.0,     -k6_y,     0.0,
            0.0,      0.0,      0.0,     -gj_l,     0.0,      0.0,      0.0,      0.0,      0.0,      gj_l,     0.0,      0.0,
            0.0,      0.0,      k6_y,     0.0,      k2_y,     0.0,      0.0,      0.0,     -k6_y,     0.0,      k4_y,     0.0,
            0.0,      k6_z,     0.0,      0.0,      0.0,      k2_z,     0.0,     -k6_z,     0.0,      0.0,      0.0,      k4_z,
        ]);

        k_local
    }

    pub fn transformation_matrix(
        &self,
        node_i: &Point3D,
        node_j: &Point3D,
    ) -> SMatrix<f64, 12, 12> {
        let dx = node_j.x - node_i.x;
        let dy = node_j.y - node_i.y;
        let dz = node_j.z - node_i.z;
        let l = self.length;

        let cx = dx / l;
        let cy = dy / l;
        let cz = dz / l;

        let tol = 1e-6;
        let r3 = if (cx.abs() - 1.0).abs() < tol {
            if cx > 0.0 {
                nalgebra::Matrix3::identity()
            } else {
                nalgebra::Matrix3::from_row_slice(&[
                    -1.0, 0.0, 0.0,
                    0.0, -1.0, 0.0,
                    0.0, 0.0, 1.0,
                ])
            }
        } else if (cy.abs() - 1.0).abs() < tol {
            if cy > 0.0 {
                nalgebra::Matrix3::from_row_slice(&[
                    0.0, 1.0, 0.0,
                    -1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0,
                ])
            } else {
                nalgebra::Matrix3::from_row_slice(&[
                    0.0, -1.0, 0.0,
                    1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0,
                ])
            }
        } else if (cz.abs() - 1.0).abs() < tol {
            if cz > 0.0 {
                nalgebra::Matrix3::from_row_slice(&[
                    0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0,
                    -1.0, 0.0, 0.0,
                ])
            } else {
                nalgebra::Matrix3::from_row_slice(&[
                    0.0, 0.0, -1.0,
                    0.0, 1.0, 0.0,
                    1.0, 0.0, 0.0,
                ])
            }
        } else {
            let d = (cx * cx + cy * cy).sqrt();
            nalgebra::Matrix3::from_row_slice(&[
                cx, cy, cz,
                -cy / d, cx / d, 0.0,
                -cx * cz / d, -cy * cz / d, d,
            ])
        };

        let mut t = SMatrix::<f64, 12, 12>::zeros();
        t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r3);
        t.fixed_view_mut::<3, 3>(3, 3).copy_from(&r3);
        t.fixed_view_mut::<3, 3>(6, 6).copy_from(&r3);
        t.fixed_view_mut::<3, 3>(9, 9).copy_from(&r3);

        t
    }

    pub fn global_stiffness_matrix(
        &self,
        node_i: &Point3D,
        node_j: &Point3D,
    ) -> SMatrix<f64, 12, 12> {
        let k_local = self.local_stiffness_matrix();
        let t = self.transformation_matrix(node_i, node_j);

        t.transpose() * k_local * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beam_stiffness_simple() {
        let node_i = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let node_j = Point3D {
            x: 10.0,
            y: 0.0,
            z: 0.0,
        };
        let e = 210e9;
        let a = 0.01;
        let i = 8.333e-6;

        let beam = BeamElement::new(&node_i, &node_j, e, e / (2.0 * 1.3), a, i, i, 1.0e-6);

        assert_relative_eq!(beam.length, 10.0, epsilon = 1e-9);

        let k_local = beam.local_stiffness_matrix();
        assert!(k_local[(0, 0)] > 0.0);
        assert!(k_local[(1, 1)] > 0.0);
    }
}
