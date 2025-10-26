use fe_engine::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FE Engine Large Model Benchmark ===\n");

    let num_elements = 1665;
    let num_nodes = num_elements + 1;
    let total_dof = num_nodes * 6;

    println!("Building model:");
    println!("  - Nodes: {}", num_nodes);
    println!("  - Elements: {}", num_elements);
    println!("  - Total DOF: {}\n", total_dof);

    let build_start = Instant::now();
    let mut builder = ModelBuilder::new("Large Beam Benchmark");

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

    let total_length = 1000.0;

    let node_ids: Vec<_> = (0..=num_elements)
        .map(|i| {
            let x = (i as f64) * total_length / (num_elements as f64);
            builder.add_node(Point3D { x, y: 0.0, z: 0.0 })
        })
        .collect();

    for i in 0..num_elements {
        builder.add_beam_element(node_ids[i], node_ids[i + 1], 0, section.clone())?;
    }

    builder.add_support(node_ids[0], SupportType::Fixed)?;
    builder.add_support(
        node_ids[num_elements],
        SupportType::Roller {
            free_direction: Direction::X,
        },
    )?;

    builder
        .create_load_case("Dead Load", LoadType::Dead)
        .add_uniform_load_on_all_elements(
            5_000.0,
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
        )?
        .finish();

    let model = builder.build()?;
    let build_time = build_start.elapsed();

    println!("✓ Model built in {:.2} ms\n", build_time.as_millis());

    println!("Running analysis...");
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let analysis_start = Instant::now();
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case)?;
    let total_analysis_time = analysis_start.elapsed();

    println!("\n=== Performance Results ===");
    println!(
        "Solve time:       {:.2} ms",
        result.solver_info.solve_time.as_millis()
    );
    println!(
        "Total analysis:   {:.2} ms",
        total_analysis_time.as_millis()
    );
    println!("Build time:       {:.2} ms", build_time.as_millis());
    println!(
        "Grand total:      {:.2} ms\n",
        (build_time + total_analysis_time).as_millis()
    );

    let max_disp = result.max_displacement();
    let (node_id, _) = result.max_displacement_location();
    println!("=== Results ===");
    println!(
        "Max displacement: {:.3} mm at node {}",
        max_disp * 1000.0,
        node_id
    );

    let (total_force, _) = result.total_reactions();
    println!(
        "Total reactions: {:.1} kN (vertical)",
        total_force.z.abs() / 1000.0
    );

    let target_time_ms = 5000.0;
    println!("\n=== Performance Check ===");
    if result.solver_info.solve_time.as_millis() as f64 <= target_time_ms {
        println!(
            "✓ Solve time {:.2} ms is within target of {} ms",
            result.solver_info.solve_time.as_millis(),
            target_time_ms
        );
    } else {
        println!(
            "✗ Solve time {:.2} ms exceeds target of {} ms",
            result.solver_info.solve_time.as_millis(),
            target_time_ms
        );
    }

    Ok(())
}
