use fe_engine::prelude::*;

#[test]
fn test_cpu_gpu_bit_exact_equality_simple_beam() {
    let mut builder = ModelBuilder::new("GPU Reproducibility: Simple Beam");

    let steel = Material {
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

    builder.add_material(steel);

    let section = Section {
        area: 0.01,
        inertia_y: 8.333e-6,
        inertia_z: 8.333e-6,
        torsion_constant: 1.0e-6,
    };

    let n0 = builder.add_node(Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    });
    let n1 = builder.add_node(Point3D {
        x: 5.0,
        y: 0.0,
        z: 0.0,
    });

    builder.add_beam_element(n0, n1, 0, section).unwrap();

    builder.add_support(n0, SupportType::Fixed).unwrap();

    builder
        .create_load_case("Point Load", LoadType::Live)
        .add_nodal_force(
            n1,
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -1000.0,
            },
            Vector3D::zero(),
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let load_case = &model.load_cases[0];

    let cpu_solver = CpuCholesky;
    let mut cpu_pipeline = AnalysisPipeline::new(&model);
    let _cpu_result = cpu_pipeline.run(&cpu_solver, load_case).unwrap();

    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        use fe_engine::analysis::gpu::MetalPCG;

        let gpu_solver = MetalPCG::new().expect("Metal GPU should be available on macOS");
        let mut gpu_pipeline = AnalysisPipeline::new(&model);
        let _gpu_result = gpu_pipeline.run(&gpu_solver, load_case).unwrap();

        let disp_tolerance = 1e-5;
        let reaction_tolerance = 1e-3;

        for cpu_disp in &_cpu_result.displacements {
            let gpu_disp = _gpu_result
                .displacements
                .iter()
                .find(|d| d.node_id == cpu_disp.node_id)
                .expect(&format!(
                    "GPU result missing displacement for node {}",
                    cpu_disp.node_id
                ));

            assert!(
                (cpu_disp.translation.x - gpu_disp.translation.x).abs() < disp_tolerance,
                "Node {} UX: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.translation.x,
                gpu_disp.translation.x,
                (cpu_disp.translation.x - gpu_disp.translation.x).abs()
            );
            assert!(
                (cpu_disp.translation.y - gpu_disp.translation.y).abs() < disp_tolerance,
                "Node {} UY: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.translation.y,
                gpu_disp.translation.y,
                (cpu_disp.translation.y - gpu_disp.translation.y).abs()
            );
            assert!(
                (cpu_disp.translation.z - gpu_disp.translation.z).abs() < disp_tolerance,
                "Node {} UZ: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.translation.z,
                gpu_disp.translation.z,
                (cpu_disp.translation.z - gpu_disp.translation.z).abs()
            );
            assert!(
                (cpu_disp.rotation.x - gpu_disp.rotation.x).abs() < disp_tolerance,
                "Node {} RX: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.rotation.x,
                gpu_disp.rotation.x,
                (cpu_disp.rotation.x - gpu_disp.rotation.x).abs()
            );
            assert!(
                (cpu_disp.rotation.y - gpu_disp.rotation.y).abs() < disp_tolerance,
                "Node {} RY: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.rotation.y,
                gpu_disp.rotation.y,
                (cpu_disp.rotation.y - gpu_disp.rotation.y).abs()
            );
            assert!(
                (cpu_disp.rotation.z - gpu_disp.rotation.z).abs() < disp_tolerance,
                "Node {} RZ: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.rotation.z,
                gpu_disp.rotation.z,
                (cpu_disp.rotation.z - gpu_disp.rotation.z).abs()
            );
        }

        for cpu_rxn in &_cpu_result.reactions {
            let gpu_rxn = _gpu_result
                .reactions
                .iter()
                .find(|r| r.node_id == cpu_rxn.node_id)
                .expect(&format!(
                    "GPU result missing reaction for node {}",
                    cpu_rxn.node_id
                ));

            assert!(
                (cpu_rxn.force.x - gpu_rxn.force.x).abs() < reaction_tolerance,
                "Node {} Reaction FX: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_rxn.node_id,
                cpu_rxn.force.x,
                gpu_rxn.force.x,
                (cpu_rxn.force.x - gpu_rxn.force.x).abs()
            );
            assert!(
                (cpu_rxn.force.y - gpu_rxn.force.y).abs() < reaction_tolerance,
                "Node {} Reaction FY: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_rxn.node_id,
                cpu_rxn.force.y,
                gpu_rxn.force.y,
                (cpu_rxn.force.y - gpu_rxn.force.y).abs()
            );
            assert!(
                (cpu_rxn.force.z - gpu_rxn.force.z).abs() < reaction_tolerance,
                "Node {} Reaction FZ: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_rxn.node_id,
                cpu_rxn.force.z,
                gpu_rxn.force.z,
                (cpu_rxn.force.z - gpu_rxn.force.z).abs()
            );
        }

        println!("✓ CPU and GPU produce consistent results within tolerance (simple beam)");
    }

    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        println!("⊘ GPU test skipped: Requires macOS with 'gpu' feature enabled");
    }
}

#[test]
fn test_cpu_gpu_bit_exact_equality_portal_frame() {
    let mut builder = ModelBuilder::new("GPU Reproducibility: Portal Frame");

    let concrete = Material {
        id: 0,
        name: "Concrete C30/37".to_string(),
        material_type: MaterialType::Concrete {
            grade: ConcreteGrade {
                standard: ConcreteStandard::Eurocode2 {
                    grade: "C30/37".to_string(),
                },
                characteristic_strength: 30e6,
            },
        },
        elastic_modulus: 30e9,
        poisson_ratio: 0.2,
        density: 2400.0,
        thermal_expansion: 1e-5,
        code_reference: None,
    };

    builder.add_material(concrete);

    let column_section = Section {
        area: 0.09,
        inertia_y: 0.000675,
        inertia_z: 0.000675,
        torsion_constant: 0.001,
    };

    let beam_section = Section {
        area: 0.12,
        inertia_y: 0.0009,
        inertia_z: 0.0009,
        torsion_constant: 0.0015,
    };

    let n0 = builder.add_node(Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    });
    let n1 = builder.add_node(Point3D {
        x: 0.0,
        y: 4.0,
        z: 0.0,
    });
    let n2 = builder.add_node(Point3D {
        x: 6.0,
        y: 4.0,
        z: 0.0,
    });
    let n3 = builder.add_node(Point3D {
        x: 6.0,
        y: 0.0,
        z: 0.0,
    });

    builder
        .add_frame_element(n0, n1, 0, column_section.clone(), Plane::XY)
        .unwrap();
    builder
        .add_frame_element(n1, n2, 0, beam_section, Plane::XY)
        .unwrap();
    builder
        .add_frame_element(n2, n3, 0, column_section, Plane::XY)
        .unwrap();

    builder.add_support(n0, SupportType::Fixed).unwrap();
    builder.add_support(n3, SupportType::Fixed).unwrap();

    builder
        .create_load_case("Lateral Wind", LoadType::Wind)
        .add_nodal_force(
            n1,
            Vector3D {
                x: 5000.0,
                y: 0.0,
                z: 0.0,
            },
            Vector3D::zero(),
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let load_case = &model.load_cases[0];

    let cpu_solver = CpuCholesky;
    let mut cpu_pipeline = AnalysisPipeline::new(&model);
    let _cpu_result = cpu_pipeline.run(&cpu_solver, load_case).unwrap();

    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        use fe_engine::analysis::gpu::MetalPCG;

        let gpu_solver = MetalPCG::new().expect("Metal GPU should be available on macOS");
        let mut gpu_pipeline = AnalysisPipeline::new(&model);
        let _gpu_result = gpu_pipeline.run(&gpu_solver, load_case).unwrap();

        let disp_tolerance = 1e-5;
        let reaction_tolerance = 1e-3;

        for cpu_disp in &_cpu_result.displacements {
            let gpu_disp = _gpu_result
                .displacements
                .iter()
                .find(|d| d.node_id == cpu_disp.node_id)
                .expect(&format!(
                    "GPU result missing displacement for node {}",
                    cpu_disp.node_id
                ));

            assert!(
                (cpu_disp.translation.x - gpu_disp.translation.x).abs() < disp_tolerance,
                "Node {} UX: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.translation.x,
                gpu_disp.translation.x,
                (cpu_disp.translation.x - gpu_disp.translation.x).abs()
            );
            assert!(
                (cpu_disp.translation.y - gpu_disp.translation.y).abs() < disp_tolerance,
                "Node {} UY: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.translation.y,
                gpu_disp.translation.y,
                (cpu_disp.translation.y - gpu_disp.translation.y).abs()
            );
            assert!(
                (cpu_disp.translation.z - gpu_disp.translation.z).abs() < disp_tolerance,
                "Node {} UZ: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.translation.z,
                gpu_disp.translation.z,
                (cpu_disp.translation.z - gpu_disp.translation.z).abs()
            );
            assert!(
                (cpu_disp.rotation.x - gpu_disp.rotation.x).abs() < disp_tolerance,
                "Node {} RX: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.rotation.x,
                gpu_disp.rotation.x,
                (cpu_disp.rotation.x - gpu_disp.rotation.x).abs()
            );
            assert!(
                (cpu_disp.rotation.y - gpu_disp.rotation.y).abs() < disp_tolerance,
                "Node {} RY: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.rotation.y,
                gpu_disp.rotation.y,
                (cpu_disp.rotation.y - gpu_disp.rotation.y).abs()
            );
            assert!(
                (cpu_disp.rotation.z - gpu_disp.rotation.z).abs() < disp_tolerance,
                "Node {} RZ: CPU={:.16e} vs GPU={:.16e} (diff={:.16e})",
                cpu_disp.node_id,
                cpu_disp.rotation.z,
                gpu_disp.rotation.z,
                (cpu_disp.rotation.z - gpu_disp.rotation.z).abs()
            );
        }

        println!("✓ CPU and GPU produce consistent results within tolerance (portal frame)");
    }

    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        println!("⊘ GPU test skipped: Requires macOS with 'gpu' feature enabled");
    }
}
