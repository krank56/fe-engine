use approx::assert_relative_eq;
use fe_engine::prelude::*;

#[test]
fn test_portal_frame_lateral_load() {
    let mut builder = ModelBuilder::new("Portal Frame with Lateral Load");

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

    let column_section = Section {
        area: 0.01,
        inertia_y: 8.33e-5,
        inertia_z: 8.33e-5,
        torsion_constant: 1.0e-6,
    };

    let beam_section = Section {
        area: 0.012,
        inertia_y: 1.0e-4,
        inertia_z: 1.0e-4,
        torsion_constant: 1.0e-6,
    };

    let height = 4.0;
    let span = 6.0;

    let n0 = builder.add_node(Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    });
    let n1 = builder.add_node(Point3D {
        x: 0.0,
        y: 0.0,
        z: height,
    });
    let n2 = builder.add_node(Point3D {
        x: span,
        y: 0.0,
        z: height,
    });
    let n3 = builder.add_node(Point3D {
        x: span,
        y: 0.0,
        z: 0.0,
    });

    builder
        .add_frame_element(n0, n1, 0, column_section.clone(), Plane::XZ)
        .unwrap();
    builder
        .add_frame_element(n1, n2, 0, beam_section.clone(), Plane::XZ)
        .unwrap();
    builder
        .add_frame_element(n2, n3, 0, column_section.clone(), Plane::XZ)
        .unwrap();

    builder.add_support(n0, SupportType::Fixed).unwrap();
    builder.add_support(n3, SupportType::Fixed).unwrap();

    let lateral_force = 10000.0;

    builder
        .create_load_case("Lateral Load", LoadType::Wind)
        .add_nodal_force(
            n1,
            Vector3D {
                x: lateral_force,
                y: 0.0,
                z: 0.0,
            },
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();

    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);

    let result = pipeline.run(&solver, load_case).unwrap();

    let disp_n1 = result.displacement_at_node(n1).unwrap();

    assert!(disp_n1.translation.x > 0.0);

    let (total_force, _total_moment) = result.total_reactions();
    
    assert_relative_eq!(
        total_force.x.abs(),
        lateral_force,
        epsilon = lateral_force * 0.01
    );

    assert!(result.verify_equilibrium(load_case, &model));
}

#[test]
fn test_portal_frame_vertical_load() {
    let mut builder = ModelBuilder::new("Portal Frame with Vertical Load");

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
        elastic_modulus: 33e9,
        poisson_ratio: 0.2,
        density: 2500.0,
        thermal_expansion: 10e-6,
        code_reference: None,
    };

    builder.add_material(concrete);

    let column_section = Section {
        area: 0.09,
        inertia_y: 6.75e-4,
        inertia_z: 6.75e-4,
        torsion_constant: 1.0e-6,
    };

    let beam_section = Section {
        area: 0.12,
        inertia_y: 9.0e-4,
        inertia_z: 9.0e-4,
        torsion_constant: 1.0e-6,
    };

    let height = 5.0;
    let span = 8.0;
    let num_beam_elements = 4;

    let n0 = builder.add_node(Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    });

    let beam_nodes: Vec<_> = (0..=num_beam_elements)
        .map(|i| {
            let x = (i as f64) * span / (num_beam_elements as f64);
            builder.add_node(Point3D {
                x,
                y: 0.0,
                z: height,
            })
        })
        .collect();

    let n3 = builder.add_node(Point3D {
        x: span,
        y: 0.0,
        z: 0.0,
    });

    builder
        .add_frame_element(n0, beam_nodes[0], 0, column_section.clone(), Plane::XZ)
        .unwrap();

    let mut beam_element_ids = Vec::new();
    for i in 0..num_beam_elements {
        let elem_id = builder
            .add_frame_element(
                beam_nodes[i],
                beam_nodes[i + 1],
                0,
                beam_section.clone(),
                Plane::XZ,
            )
            .unwrap();
        beam_element_ids.push(elem_id);
    }

    builder
        .add_frame_element(
            beam_nodes[num_beam_elements],
            n3,
            0,
            column_section.clone(),
            Plane::XZ,
        )
        .unwrap();

    builder.add_support(n0, SupportType::Fixed).unwrap();
    builder.add_support(n3, SupportType::Fixed).unwrap();

    let udl = 15000.0;

    let load_case_builder = builder.create_load_case("Uniform Dead Load", LoadType::Dead);
    
    for &elem_id in &beam_element_ids {
        load_case_builder
            .add_element_load(
                elem_id,
                LoadDistribution::UniformDistributed {
                    intensity: udl,
                    direction: Vector3D {
                        x: 0.0,
                        y: 0.0,
                        z: -1.0,
                    },
                },
            )
            .unwrap();
    }
    
    load_case_builder.finish();

    let model = builder.build().unwrap();

    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);

    let result = pipeline.run(&solver, load_case).unwrap();

    let mid_node = beam_nodes[num_beam_elements / 2];
    let mid_disp = result.displacement_at_node(mid_node).unwrap();

    assert!(mid_disp.translation.z < 0.0);

    let (total_force, _) = result.total_reactions();
    let expected_total_force = udl * span;
    assert_relative_eq!(
        total_force.z.abs(),
        expected_total_force,
        epsilon = expected_total_force * 0.01
    );

    assert!(result.verify_equilibrium(load_case, &model));
}
