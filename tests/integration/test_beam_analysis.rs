use approx::assert_relative_eq;
use fe_engine::prelude::*;

#[test]
fn test_end_to_end_beam_analysis() {
    let mut builder = ModelBuilder::new("Integration Test: Complete Beam Analysis Workflow");

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

    let width: f64 = 0.3;
    let height: f64 = 0.5;
    let area = width * height;
    let inertia = width * height.powi(3) / 12.0;

    let section = Section {
        area,
        inertia_y: inertia,
        inertia_z: inertia,
        torsion_constant: 1.0e-6,
    };

    let span = 8.0;
    let num_elements = 16;

    let nodes: Vec<_> = (0..=num_elements)
        .map(|i| {
            let x = (i as f64) * span / (num_elements as f64);
            builder.add_node(Point3D { x, y: 0.0, z: 0.0 })
        })
        .collect();

    for i in 0..num_elements {
        builder
            .add_beam_element(nodes[i], nodes[i + 1], 0, section.clone())
            .unwrap();
    }

    builder.add_support(nodes[0], SupportType::Fixed).unwrap();
    builder
        .add_support(
            nodes[num_elements],
            SupportType::Roller {
                free_direction: Direction::X,
            },
        )
        .unwrap();

    let udl = 15000.0;

    builder
        .create_load_case("UDL", LoadType::Live)
        .add_uniform_load_on_all_elements(
            udl,
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
        )
        .unwrap()
        .finish();

    builder
        .create_load_case("Point Load", LoadType::Live)
        .add_nodal_force(
            nodes[num_elements / 2],
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -50000.0,
            },
            Vector3D::zero(),
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();

    let solver = CpuCholesky;

    for load_case in model.load_cases.iter() {
        let mut pipeline = AnalysisPipeline::new(&model);
        let result = pipeline.run(&solver, load_case).unwrap();

        assert!(!result.displacements.is_empty());
        assert!(!result.reactions.is_empty());
        assert!(!result.element_forces.is_empty());

        let max_disp = result.max_displacement();
        assert!(max_disp > 0.0);
        assert!(max_disp < span / 100.0);

        let (total_force, _) = result.total_reactions();

        if load_case.name == "UDL" {
            let expected_total = udl * span;
            assert_relative_eq!(
                total_force.z.abs(),
                expected_total,
                epsilon = expected_total * 0.01
            );
        } else if load_case.name == "Point Load" {
            assert_relative_eq!(total_force.z.abs(), 50000.0, epsilon = 500.0);
        }

        let fixed_disp = result.displacement_at_node(nodes[0]).unwrap();
        assert_relative_eq!(fixed_disp.translation.magnitude(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fixed_disp.rotation.magnitude(), 0.0, epsilon = 1e-10);

        assert!(result.to_json().is_ok());
    }
}

