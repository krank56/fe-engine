use approx::assert_relative_eq;
use fe_engine::prelude::*;

#[test]
fn test_simple_beam_deflection() {
    let mut builder = ModelBuilder::new("Simple Beam Deflection Test");

    let mat = Material {
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

    builder.add_material(mat);

    let b: f64 = 0.3;
    let h: f64 = 0.3;
    let area = b * h;
    let inertia = b * h.powi(3) / 12.0;

    let section = Section {
        area,
        inertia_y: inertia,
        inertia_z: inertia,
        torsion_constant: 1.0e-6,
    };

    let length = 10.0;
    let num_elements = 20;

    let node_ids: Vec<_> = (0..=num_elements)
        .map(|i| {
            let x = (i as f64) * length / (num_elements as f64);
            builder.add_node(Point3D { x, y: 0.0, z: 0.0 })
        })
        .collect();

    for i in 0..num_elements {
        builder
            .add_beam_element(node_ids[i], node_ids[i + 1], 0, section.clone())
            .unwrap();
    }

    builder
        .add_support(node_ids[0], SupportType::Pinned)
        .unwrap();
    builder
        .add_support(
            node_ids[num_elements],
            SupportType::Roller {
                free_direction: Direction::X,
            },
        )
        .unwrap();

    let q = 10_000.0;

    builder
        .create_load_case("Uniform Load", LoadType::Live)
        .add_uniform_load_on_all_elements(
            q,
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();

    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    
    let result = pipeline.run(&solver, load_case).unwrap();

    let mid_node = num_elements / 2;
    let mid_disp = result.displacement_at_node(mid_node).unwrap();

    let e = 33e9;
    let i = inertia;
    let l = length;
    let expected_max_deflection = (5.0 * q * l.powi(4)) / (384.0 * e * i);

    assert_relative_eq!(
        mid_disp.translation.z.abs(),
        expected_max_deflection,
        epsilon = expected_max_deflection * 0.01
    );

    let (total_force, _) = result.total_reactions();
    let expected_total_force = q * l;
    assert_relative_eq!(
        total_force.z.abs(),
        expected_total_force,
        epsilon = expected_total_force * 0.01
    );

    assert!(result.check_deflection_limit(150.0, length));
}
