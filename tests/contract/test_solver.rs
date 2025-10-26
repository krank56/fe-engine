use approx::assert_relative_eq;
use fe_engine::prelude::*;
use fe_engine::analysis::{auto_select_solver, create_solver, is_gpu_available, SolverBackend};

#[test]
fn test_solver_satisfies_ku_equals_f() {
    let mut builder = ModelBuilder::new("Contract Test: K*u = F");

    let mat = Material {
        id: 0,
        name: "Steel S355".to_string(),
        material_type: MaterialType::Steel {
            grade: SteelGrade {
                standard: SteelStandard::Eurocode3 {
                    grade: "S355".to_string(),
                },
                yield_strength: 355e6,
            },
        },
        elastic_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        thermal_expansion: 12e-6,
        code_reference: None,
    };

    builder.add_material(mat);

    let section = Section {
        area: 0.01,
        inertia_y: 8.333e-6,
        inertia_z: 8.333e-6,
        torsion_constant: 1.0e-6,
    };

    let n0 = builder.add_node(Point3D { x: 0.0, y: 0.0, z: 0.0 });
    let n1 = builder.add_node(Point3D { x: 5.0, y: 0.0, z: 0.0 });

    builder.add_beam_element(n0, n1, 0, section).unwrap();

    builder.add_support(n0, SupportType::Fixed).unwrap();

    builder
        .create_load_case("Point Load", LoadType::Live)
        .add_nodal_force(
            n1,
            Vector3D { x: 0.0, y: 0.0, z: -1000.0 },
            Vector3D::zero(),
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let (total_force, total_moment) = result.total_reactions();

    assert_relative_eq!(total_force.x, 0.0, epsilon = 1e-6);
    assert_relative_eq!(total_force.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(total_force.z.abs(), 1000.0, epsilon = 1e-6);

    let expected_moment = 1000.0 * 5.0;
    assert_relative_eq!(total_moment.y.abs(), expected_moment, epsilon = expected_moment * 1e-6);
}

#[test]
fn test_displacement_boundary_conditions_enforced() {
    let mut builder = ModelBuilder::new("Contract Test: BC Enforcement");

    let mat = Material {
        id: 0,
        name: "Concrete".to_string(),
        material_type: MaterialType::Concrete {
            grade: ConcreteGrade {
                standard: ConcreteStandard::Eurocode2 { grade: "C30/37".to_string() },
                characteristic_strength: 30e6,
            },
        },
        elastic_modulus: 33e9,
        poisson_ratio: 0.2,
        density: 2500.0,
        thermal_expansion: 10e-6,
        code_reference: None,
    };

    builder.add_material(mat);

    let section = Section {
        area: 0.09,
        inertia_y: 6.75e-4,
        inertia_z: 6.75e-4,
        torsion_constant: 1.0e-6,
    };

    let n0 = builder.add_node(Point3D { x: 0.0, y: 0.0, z: 0.0 });
    let n1 = builder.add_node(Point3D { x: 10.0, y: 0.0, z: 0.0 });

    builder.add_beam_element(n0, n1, 0, section).unwrap();

    builder.add_support(n0, SupportType::Fixed).unwrap();

    builder
        .create_load_case("Load", LoadType::Live)
        .add_nodal_force(n1, Vector3D { x: 1000.0, y: 0.0, z: 0.0 }, Vector3D::zero())
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let fixed_node_disp = result.displacement_at_node(n0).unwrap();

    assert_relative_eq!(fixed_node_disp.translation.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(fixed_node_disp.translation.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(fixed_node_disp.translation.z, 0.0, epsilon = 1e-12);
    assert_relative_eq!(fixed_node_disp.rotation.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(fixed_node_disp.rotation.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(fixed_node_disp.rotation.z, 0.0, epsilon = 1e-12);
}

#[test]
fn test_auto_select_solver_small_model() {
    let backend = auto_select_solver(100);
    assert_eq!(backend, SolverBackend::CpuCholesky);
}

#[test]
fn test_auto_select_solver_large_model() {
    let backend = auto_select_solver(10000);
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        if is_gpu_available() {
            assert_eq!(backend, SolverBackend::GpuIterative);
        } else {
            assert_eq!(backend, SolverBackend::CpuCholesky);
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        assert_eq!(backend, SolverBackend::CpuCholesky);
    }
}

#[test]
fn test_create_solver_cpu() {
    let solver = create_solver(SolverBackend::CpuCholesky);
    assert_eq!(solver.name(), "CPU Cholesky (nalgebra-sparse)");
}

#[test]
#[cfg(all(target_os = "macos", feature = "gpu"))]
fn test_create_solver_gpu_fallback() {
    let solver = create_solver(SolverBackend::GpuIterative);
    if is_gpu_available() {
        assert_eq!(solver.name(), "Metal PCG (GPU)");
    } else {
        assert_eq!(solver.name(), "CPU Cholesky (nalgebra-sparse)");
    }
}

#[test]
fn test_is_gpu_available() {
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        let available = is_gpu_available();
        println!("GPU available: {}", available);
    }
    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        assert!(!is_gpu_available());
    }
}
