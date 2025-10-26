use fe_engine::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FE Engine Quickstart Validation ===\n");

    let mut builder = ModelBuilder::new("Simple Beam Test");

    let mat = Material {
        id: 0,
        name: "Concrete C30".to_string(),
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

    let section = Section {
        area: 0.09,
        inertia_y: 6.75e-4,
        inertia_z: 6.75e-4,
        torsion_constant: 1.0e-4,
    };

    let length = 10.0;
    let num_elements = 10;

    let node_ids: Vec<_> = (0..=num_elements)
        .map(|i| {
            let x = (i as f64) * length / (num_elements as f64);
            builder.add_node(Point3D { x, y: 0.0, z: 0.0 })
        })
        .collect();

    for i in 0..num_elements {
        builder.add_beam_element(node_ids[i], node_ids[i + 1], 0, section.clone())?;
    }

    builder.add_support(node_ids[0], SupportType::Pinned)?;
    builder.add_support(
        node_ids[num_elements],
        SupportType::Roller {
            free_direction: Direction::X,
        },
    )?;

    builder
        .create_load_case("Dead Load", LoadType::Dead)
        .add_uniform_load_on_all_elements(
            10_000.0,
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
        )?
        .finish();

    let model = builder.build()?;
    println!(
        "✓ Model created: {} nodes, {} elements",
        model.nodes.len(),
        model.elements.len()
    );

    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);

    let result = pipeline.run(&solver, load_case)?;

    println!(
        "✓ Analysis complete in {:.2} ms",
        result.solver_info.solve_time.as_millis()
    );

    let max_disp = result.max_displacement();
    let (node_id, _) = result.max_displacement_location();
    println!(
        "Max displacement: {:.3} mm at node {}",
        max_disp * 1000.0,
        node_id
    );

    if result.check_deflection_limit(250.0, length) {
        println!(
            "✓ Deflection within EC2 limit (L/250 = {:.1} mm)",
            length * 1000.0 / 250.0
        );
    } else {
        println!("✗ Deflection exceeds limit!");
    }

    let (total_force, _) = result.total_reactions();
    println!(
        "Total reactions: {:.1} kN (vertical)",
        total_force.z.abs() / 1000.0
    );

    println!("\n=== Audit Trail ===");
    println!("Total entries: {}", pipeline.audit_trail.entries.len());
    for entry in pipeline.audit_trail.entries.iter().take(5) {
        println!("  - {}", entry.action);
    }

    Ok(())
}
