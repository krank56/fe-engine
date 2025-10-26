use nalgebra::SMatrix;

use crate::structure::element::Plane;
use crate::structure::geometry::Point3D;

pub struct Frame2DElement {
    pub length: f64,
    pub area: f64,
    pub elastic_modulus: f64,
    pub inertia: f64,
    pub plane: Plane,
}

impl Frame2DElement {
    pub fn new(
        node_i: &Point3D,
        node_j: &Point3D,
        elastic_modulus: f64,
        area: f64,
        inertia: f64,
        plane: Plane,
    ) -> Self {
        let (dx, dy) = match plane {
            Plane::XY => (node_j.x - node_i.x, node_j.y - node_i.y),
            Plane::XZ => (node_j.x - node_i.x, node_j.z - node_i.z),
            Plane::YZ => (node_j.y - node_i.y, node_j.z - node_i.z),
        };

        let length = (dx * dx + dy * dy).sqrt();

        Self {
            length,
            area,
            elastic_modulus,
            inertia,
            plane,
        }
    }

    pub fn local_stiffness_matrix(&self) -> SMatrix<f64, 6, 6> {
        let e = self.elastic_modulus;
        let a = self.area;
        let l = self.length;
        let i = self.inertia;

        let ea_l = (e * a) / l;
        let ei_l3 = (e * i) / (l * l * l);

        let k12 = 12.0 * ei_l3;
        let k6 = 6.0 * ei_l3 * l;
        let k4 = 4.0 * ei_l3 * l * l;
        let k2 = 2.0 * ei_l3 * l * l;

        #[rustfmt::skip]
        let k_local = SMatrix::<f64, 6, 6>::from_row_slice(&[
            ea_l,     0.0,      0.0,     -ea_l,     0.0,      0.0,
            0.0,      k12,      k6,       0.0,     -k12,      k6,
            0.0,      k6,       k4,       0.0,     -k6,       k2,
           -ea_l,     0.0,      0.0,      ea_l,     0.0,      0.0,
            0.0,     -k12,     -k6,       0.0,      k12,     -k6,
            0.0,      k6,       k2,       0.0,     -k6,       k4,
        ]);

        k_local
    }

    pub fn transformation_matrix(&self, node_i: &Point3D, node_j: &Point3D) -> SMatrix<f64, 6, 6> {
        let (dx, dy) = match self.plane {
            Plane::XY => (node_j.x - node_i.x, node_j.y - node_i.y),
            Plane::XZ => (node_j.x - node_i.x, node_j.z - node_i.z),
            Plane::YZ => (node_j.y - node_i.y, node_j.z - node_i.z),
        };

        let l = self.length;
        let cos_theta = dx / l;
        let sin_theta = dy / l;

        let r = nalgebra::Matrix3::from_row_slice(&[
            cos_theta,
            sin_theta,
            0.0,
            -sin_theta,
            cos_theta,
            0.0,
            0.0,
            0.0,
            1.0,
        ]);

        let mut t = SMatrix::<f64, 6, 6>::zeros();
        t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        t.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);

        t
    }

    pub fn global_stiffness_matrix(
        &self,
        node_i: &Point3D,
        node_j: &Point3D,
    ) -> SMatrix<f64, 6, 6> {
        let k_local = self.local_stiffness_matrix();
        let t = self.transformation_matrix(node_i, node_j);

        t.transpose() * k_local * t
    }

    pub fn dof_mapping(&self) -> [usize; 6] {
        match self.plane {
            Plane::XY => [0, 1, 5, 0, 1, 5],
            Plane::XZ => [0, 2, 4, 0, 2, 4],
            Plane::YZ => [1, 2, 3, 1, 2, 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_frame2d_stiffness_xz_plane() {
        let node_i = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let node_j = Point3D {
            x: 0.0,
            y: 0.0,
            z: 4.0,
        };

        let e = 210e9;
        let a = 0.01;
        let i = 8.33e-5;

        let frame = Frame2DElement::new(&node_i, &node_j, e, a, i, Plane::XZ);

        assert_relative_eq!(frame.length, 4.0, epsilon = 1e-9);

        let k_local = frame.local_stiffness_matrix();

        assert!(k_local[(0, 0)] > 0.0);
        assert!(k_local[(1, 1)] > 0.0);
        assert_relative_eq!(k_local[(0, 0)], k_local[(3, 3)], epsilon = 1e-6);
        assert_relative_eq!(k_local[(0, 3)], -k_local[(0, 0)], epsilon = 1e-6);
    }

    #[test]
    fn test_frame2d_transformation_horizontal() {
        let node_i = Point3D {
            x: 0.0,
            y: 0.0,
            z: 5.0,
        };
        let node_j = Point3D {
            x: 6.0,
            y: 0.0,
            z: 5.0,
        };

        let e = 210e9;
        let a = 0.012;
        let i = 1.0e-4;

        let frame = Frame2DElement::new(&node_i, &node_j, e, a, i, Plane::XZ);

        assert_relative_eq!(frame.length, 6.0, epsilon = 1e-9);

        let t = frame.transformation_matrix(&node_i, &node_j);

        assert_relative_eq!(t[(0, 0)], 1.0, epsilon = 1e-9);
        assert_relative_eq!(t[(0, 1)], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_frame2d_dof_mapping() {
        let node_i = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let node_j = Point3D {
            x: 4.0,
            y: 0.0,
            z: 0.0,
        };

        let frame = Frame2DElement::new(&node_i, &node_j, 210e9, 0.01, 8e-5, Plane::XZ);

        let dof_map = frame.dof_mapping();

        assert_eq!(dof_map, [0, 2, 4, 0, 2, 4]);
    }
}
