use approx::assert_relative_eq;
use fe_engine::prelude::*;

#[test]
fn test_end_to_end_frame_analysis() {
    let mut builder = ModelBuilder::new("Integration Test: Multi-Element Frame Workflow");

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
    let num_beam_elements = 3;

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
    let lateral_force = 5000.0;

    let load_case_builder = builder.create_load_case("Dead Load", LoadType::Dead);
    
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

    builder
        .create_load_case("Wind Load", LoadType::Wind)
        .add_nodal_force(
            beam_nodes[0],
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

    assert_eq!(model.elements.len(), 2 + num_beam_elements);
    assert_eq!(model.load_cases.len(), 2);

    let solver = CpuCholesky;

    let mut pipeline = AnalysisPipeline::new(&model);

    let result_dead = pipeline.run(&solver, &model.load_cases[0]).unwrap();
    assert!(result_dead.verify_equilibrium(&model.load_cases[0], &model));

    let mid_node = beam_nodes[num_beam_elements / 2];
    let disp_dead = result_dead.displacement_at_node(mid_node).unwrap();
    assert!(disp_dead.translation.z < 0.0);

    let (force_dead, _) = result_dead.total_reactions();
    let expected_force = udl * span;
    assert_relative_eq!(
        force_dead.z.abs(),
        expected_force,
        epsilon = expected_force * 0.01
    );

    let result_wind = pipeline.run(&solver, &model.load_cases[1]).unwrap();
    assert!(result_wind.verify_equilibrium(&model.load_cases[1], &model));

    let disp_wind = result_wind.displacement_at_node(beam_nodes[0]).unwrap();
    assert!(disp_wind.translation.x > 0.0);

    let (force_wind, _) = result_wind.total_reactions();
    assert_relative_eq!(
        force_wind.x.abs(),
        lateral_force,
        epsilon = lateral_force * 0.01
    );

    let json_result = serde_json::to_string_pretty(&model).unwrap();
    assert!(json_result.contains("Multi-Element Frame Workflow"));
    assert!(json_result.contains("Dead Load"));
    assert!(json_result.contains("Wind Load"));

    let deserialized: StructuralModel = serde_json::from_str(&json_result).unwrap();
    assert_eq!(deserialized.elements.len(), model.elements.len());
    assert_eq!(deserialized.load_cases.len(), model.load_cases.len());
}
