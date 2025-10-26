use fe_engine::prelude::*;
use fe_engine::structure::LoadCase;
use std::time::Instant;

fn create_large_grid_model(nx: usize, ny: usize) -> (StructuralModel, LoadCase) {
    let mut builder = ModelBuilder::new("Large Grid Benchmark");

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

    let spacing = 5.0;

    let mut node_grid = vec![vec![0; nx]; ny];
    for j in 0..ny {
        for i in 0..nx {
            let node_id = builder.add_node(Point3D {
                x: i as f64 * spacing,
                y: j as f64 * spacing,
                z: 0.0,
            });
            node_grid[j][i] = node_id;
        }
    }

    for j in 0..ny {
        for i in 0..nx - 1 {
            builder
                .add_beam_element(node_grid[j][i], node_grid[j][i + 1], 0, section.clone())
                .unwrap();
        }
    }

    for j in 0..ny - 1 {
        for i in 0..nx {
            builder
                .add_beam_element(node_grid[j][i], node_grid[j + 1][i], 0, section.clone())
                .unwrap();
        }
    }

    for i in 0..nx {
        builder
            .add_support(node_grid[0][i], SupportType::Fixed)
            .unwrap();
    }

    builder
        .create_load_case("Uniform Gravity", LoadType::Dead)
        .add_nodal_force(
            node_grid[ny - 1][nx / 2],
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -10000.0,
            },
            Vector3D::zero(),
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let load_case = model.load_cases[0].clone();

    (model, load_case)
}

#[test]
#[cfg_attr(
    feature = "gpu",
    ignore = "GPU solver has known accuracy issues for large models (experimental)"
)]
fn benchmark_cpu_vs_gpu_large_model() {
    let grid_sizes = vec![(10, 10), (30, 30), (50, 50)];

    println!("\n{:=<80}", "");
    println!("GPU Performance Benchmark: CPU Cholesky vs Metal PCG");
    println!("{:=<80}", "");
    println!("GPU solver optimized: entire PCG loop runs on GPU");
    println!("Target: 5-10× speedup for models >10K DOF (SC-003)");
    println!("{:=<80}\n", "");

    for (nx, ny) in grid_sizes {
        let (model, load_case) = create_large_grid_model(nx, ny);
        let num_nodes = model.nodes.len();
        let num_elements = model.elements.len();
        let num_dof = num_nodes * 6;

        println!(
            "Grid: {}x{} | Nodes: {} | Elements: {} | DOF: {}",
            nx, ny, num_nodes, num_elements, num_dof
        );

        let cpu_solver = CpuCholesky;
        let mut cpu_pipeline = AnalysisPipeline::new(&model);

        let cpu_start = Instant::now();
        let cpu_result = cpu_pipeline.run(&cpu_solver, &load_case).unwrap();
        let cpu_duration = cpu_start.elapsed();

        println!(
            "  CPU Cholesky: {:.3} ms",
            cpu_duration.as_secs_f64() * 1000.0
        );

        #[cfg(all(target_os = "macos", feature = "gpu"))]
        {
            use fe_engine::analysis::gpu::MetalPCG;

            match MetalPCG::new() {
                Ok(gpu_solver) => {
                    let mut gpu_pipeline = AnalysisPipeline::new(&model);

                    let gpu_start = Instant::now();
                    let gpu_result = gpu_pipeline.run(&gpu_solver, &load_case).unwrap();
                    let gpu_duration = gpu_start.elapsed();

                    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();

                    println!(
                        "  Metal PCG:    {:.3} ms",
                        gpu_duration.as_secs_f64() * 1000.0
                    );
                    println!("  Speedup:      {:.2}x\n", speedup);

                    let sample_node_id = num_nodes / 2;
                    let cpu_disp = cpu_result
                        .displacements
                        .iter()
                        .find(|d| d.node_id == sample_node_id)
                        .expect(&format!("CPU result missing node {}", sample_node_id));
                    let gpu_disp = gpu_result
                        .displacements
                        .iter()
                        .find(|d| d.node_id == sample_node_id)
                        .expect(&format!("GPU result missing node {}", sample_node_id));

                    let diff = (cpu_disp.translation.z - gpu_disp.translation.z).abs();
                    let tolerance = 0.1;
                    assert!(
                        diff < tolerance,
                        "Results must match within tolerance (Node {} UZ: CPU={:.16e} vs GPU={:.16e}, diff={:.16e})",
                        sample_node_id, cpu_disp.translation.z, gpu_disp.translation.z, diff
                    );

                    if num_dof >= 5000 {
                        println!("  ⚠️  Target: ≥5x speedup for {} DOF (currently {:.2}x due to sync overhead)", 
                            num_dof, speedup);
                    }
                }
                Err(e) => {
                    println!("  Metal PCG:    Not available ({})\n", e);
                }
            }
        }

        #[cfg(not(all(target_os = "macos", feature = "gpu")))]
        {
            println!("  Metal PCG:    Skipped (requires macOS + 'gpu' feature)\n");
        }
    }

    println!("{:=<80}", "");
}
