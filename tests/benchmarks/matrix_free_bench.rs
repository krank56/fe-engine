use fe_engine::prelude::*;
use std::time::Instant;

#[test]
fn benchmark_simple_beam_all_solvers() {
    println!("\n{:=<80}", "");
    println!("Simple Beam Benchmark: CPU vs GPU Sparse vs GPU Matrix-Free");
    println!("{:=<80}\n", "");

    let mut builder = ModelBuilder::new("Simple Beam Benchmark");

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

    println!("Model: {} nodes, {} elements, {} DOF\n",
        model.nodes.len(), model.elements.len(), model.nodes.len() * 6);

    // CPU Solver
    let cpu_solver = CpuCholesky;
    let mut cpu_pipeline = AnalysisPipeline::new(&model);

    let cpu_start = Instant::now();
    let cpu_result = cpu_pipeline.run(&cpu_solver, load_case).unwrap();
    let cpu_duration = cpu_start.elapsed();

    println!("CPU Cholesky:     {:.3} ms", cpu_duration.as_secs_f64() * 1000.0);

    // GPU Sparse Solver
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        use fe_engine::analysis::gpu::{MetalPCG, MatrixFreeGPU};

        // Sparse GPU solver
        match MetalPCG::new() {
            Ok(gpu_solver) => {
                let mut gpu_pipeline = AnalysisPipeline::new(&model);

                let gpu_start = Instant::now();
                let gpu_result = gpu_pipeline.run(&gpu_solver, load_case).unwrap();
                let gpu_duration = gpu_start.elapsed();

                let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();

                println!("GPU Sparse PCG:   {:.3} ms ({})",
                    gpu_duration.as_secs_f64() * 1000.0,
                    if speedup < 1.0 {
                        format!("{:.2}x SLOWER", 1.0/speedup)
                    } else {
                        format!("{:.2}x faster", speedup)
                    });

                // Verify results match
                let cpu_uz = cpu_result.displacements[1].translation.z;
                let gpu_uz = gpu_result.displacements[1].translation.z;
                let diff = (cpu_uz - gpu_uz).abs();
                println!("  Result diff:    {:.3e} m ({:.3}% error)", diff, (diff / cpu_uz.abs()) * 100.0);
            }
            Err(e) => {
                println!("GPU Sparse PCG:   Not available ({})", e);
            }
        }

        // Matrix-Free GPU solver
        match MatrixFreeGPU::new(&model) {
            Ok(mf_solver) => {
                let mut mf_pipeline = AnalysisPipeline::new(&model);

                let mf_start = Instant::now();
                let mf_result = mf_pipeline.run(&mf_solver, load_case).unwrap();
                let mf_duration = mf_start.elapsed();

                let speedup = cpu_duration.as_secs_f64() / mf_duration.as_secs_f64();

                println!("GPU Matrix-Free:  {:.3} ms ({})",
                    mf_duration.as_secs_f64() * 1000.0,
                    if speedup < 1.0 {
                        format!("{:.2}x SLOWER", 1.0/speedup)
                    } else {
                        format!("{:.2}x faster", speedup)
                    });

                // Verify results match
                let cpu_uz = cpu_result.displacements[1].translation.z;
                let mf_uz = mf_result.displacements[1].translation.z;
                let diff = (cpu_uz - mf_uz).abs();
                println!("  CPU uz:         {:.6e} m", cpu_uz);
                println!("  Matrix-Free uz: {:.6e} m", mf_uz);
                println!("  Result diff:    {:.3e} m ({:.3}% error)", diff, (diff / cpu_uz.abs()) * 100.0);

                assert!(diff < 1e-4, "Matrix-free results must match CPU within tolerance");
            }
            Err(e) => {
                println!("GPU Matrix-Free:  Not available ({})", e);
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        println!("GPU Solvers:      Skipped (requires macOS + 'gpu' feature)");
    }

    println!("\n{:=<80}", "");
}

fn create_grid_model(nx: usize, ny: usize) -> (StructuralModel, fe_engine::structure::LoadCase) {
    let mut builder = ModelBuilder::new(&format!("Grid {}x{}", nx, ny));

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
            builder.add_beam_element(node_grid[j][i], node_grid[j][i + 1], 0, section.clone()).unwrap();
        }
    }

    for j in 0..ny - 1 {
        for i in 0..nx {
            builder.add_beam_element(node_grid[j][i], node_grid[j + 1][i], 0, section.clone()).unwrap();
        }
    }

    for i in 0..nx {
        builder.add_support(node_grid[0][i], SupportType::Fixed).unwrap();
    }

    builder
        .create_load_case("Concentrated Load", LoadType::Live)
        .add_nodal_force(
            node_grid[ny - 1][nx / 2],
            Vector3D { x: 0.0, y: 0.0, z: -10000.0 },
            Vector3D::zero(),
        )
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let load_case = model.load_cases[0].clone();

    (model, load_case)
}

#[test]
fn benchmark_medium_grid_all_solvers() {
    println!("\n{:=<80}", "");
    println!("Medium Grid (10x10) Benchmark: CPU vs GPU Sparse vs GPU Matrix-Free");
    println!("{:=<80}\n", "");

    let (model, load_case) = create_grid_model(10, 10);

    println!("Model: {} nodes, {} elements, {} DOF\n",
        model.nodes.len(), model.elements.len(), model.nodes.len() * 6);

    // CPU Solver
    let cpu_solver = CpuCholesky;
    let mut cpu_pipeline = AnalysisPipeline::new(&model);

    let cpu_start = Instant::now();
    let cpu_result = cpu_pipeline.run(&cpu_solver, &load_case).unwrap();
    let cpu_duration = cpu_start.elapsed();

    println!("CPU Cholesky:     {:.3} ms", cpu_duration.as_secs_f64() * 1000.0);

    // GPU Sparse Solver
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        use fe_engine::analysis::gpu::{MetalPCG, MatrixFreeGPU};

        // Sparse GPU solver
        match MetalPCG::new() {
            Ok(gpu_solver) => {
                let mut gpu_pipeline = AnalysisPipeline::new(&model);

                let gpu_start = Instant::now();
                let gpu_result = gpu_pipeline.run(&gpu_solver, &load_case).unwrap();
                let gpu_duration = gpu_start.elapsed();

                let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();

                println!("GPU Sparse PCG:   {:.3} ms ({})",
                    gpu_duration.as_secs_f64() * 1000.0,
                    if speedup < 1.0 {
                        format!("{:.2}x SLOWER", 1.0/speedup)
                    } else {
                        format!("{:.2}x faster", speedup)
                    });

                // Verify results match
                let sample_node_id = model.nodes.len() / 2;
                let cpu_uz = cpu_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let gpu_uz = gpu_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let diff = (cpu_uz - gpu_uz).abs();
                println!("  Result diff:    {:.3e} m ({:.3}% error)", diff, (diff / cpu_uz.abs()) * 100.0);
            }
            Err(e) => {
                println!("GPU Sparse PCG:   Not available ({})", e);
            }
        }

        // Matrix-Free GPU solver
        match MatrixFreeGPU::new(&model) {
            Ok(mf_solver) => {
                let mut mf_pipeline = AnalysisPipeline::new(&model);

                let mf_start = Instant::now();
                let mf_result = mf_pipeline.run(&mf_solver, &load_case).unwrap();
                let mf_duration = mf_start.elapsed();

                let speedup = cpu_duration.as_secs_f64() / mf_duration.as_secs_f64();

                println!("GPU Matrix-Free:  {:.3} ms ({})",
                    mf_duration.as_secs_f64() * 1000.0,
                    if speedup < 1.0 {
                        format!("{:.2}x SLOWER", 1.0/speedup)
                    } else {
                        format!("{:.2}x faster", speedup)
                    });

                // Verify results match
                let sample_node_id = model.nodes.len() / 2;
                let cpu_uz = cpu_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let mf_uz = mf_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let diff = (cpu_uz - mf_uz).abs();
                println!("  Result diff:    {:.3e} m ({:.3}% error)", diff, (diff / cpu_uz.abs()) * 100.0);

                assert!(diff < 1e-3, "Matrix-free results must match CPU within tolerance");
            }
            Err(e) => {
                println!("GPU Matrix-Free:  Not available ({})", e);
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        println!("GPU Solvers:      Skipped (requires macOS + 'gpu' feature)");
    }

    println!("\n{:=<80}", "");
}

#[test]
fn benchmark_large_grid_all_solvers() {
    println!("\n{:=<80}", "");
    println!("Large Grid (30x30) Benchmark: CPU vs GPU Sparse vs GPU Matrix-Free");
    println!("{:=<80}\n", "");

    let (model, load_case) = create_grid_model(30, 30);

    println!("Model: {} nodes, {} elements, {} DOF\n",
        model.nodes.len(), model.elements.len(), model.nodes.len() * 6);

    // CPU Solver
    let cpu_solver = CpuCholesky;
    let mut cpu_pipeline = AnalysisPipeline::new(&model);

    let cpu_start = Instant::now();
    let cpu_result = cpu_pipeline.run(&cpu_solver, &load_case).unwrap();
    let cpu_duration = cpu_start.elapsed();

    println!("CPU Cholesky:     {:.3} ms", cpu_duration.as_secs_f64() * 1000.0);

    // GPU Sparse Solver
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    {
        use fe_engine::analysis::gpu::{MetalPCG, MatrixFreeGPU};

        // Sparse GPU solver
        match MetalPCG::new() {
            Ok(gpu_solver) => {
                let mut gpu_pipeline = AnalysisPipeline::new(&model);

                let gpu_start = Instant::now();
                let gpu_result = gpu_pipeline.run(&gpu_solver, &load_case).unwrap();
                let gpu_duration = gpu_start.elapsed();

                let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();

                println!("GPU Sparse PCG:   {:.3} ms ({})",
                    gpu_duration.as_secs_f64() * 1000.0,
                    if speedup < 1.0 {
                        format!("{:.2}x SLOWER", 1.0/speedup)
                    } else {
                        format!("{:.2}x faster", speedup)
                    });

                // Verify results match
                let sample_node_id = model.nodes.len() / 2;
                let cpu_uz = cpu_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let gpu_uz = gpu_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let diff = (cpu_uz - gpu_uz).abs();
                println!("  Result diff:    {:.3e} m ({:.3}% error)", diff, (diff / cpu_uz.abs()) * 100.0);
            }
            Err(e) => {
                println!("GPU Sparse PCG:   Not available ({})", e);
            }
        }

        // Matrix-Free GPU solver
        match MatrixFreeGPU::new(&model) {
            Ok(mf_solver) => {
                let mut mf_pipeline = AnalysisPipeline::new(&model);

                let mf_start = Instant::now();
                let mf_result = mf_pipeline.run(&mf_solver, &load_case).unwrap();
                let mf_duration = mf_start.elapsed();

                let speedup = cpu_duration.as_secs_f64() / mf_duration.as_secs_f64();

                println!("GPU Matrix-Free:  {:.3} ms ({})",
                    mf_duration.as_secs_f64() * 1000.0,
                    if speedup < 1.0 {
                        format!("{:.2}x SLOWER", 1.0/speedup)
                    } else {
                        format!("{:.2}x faster", speedup)
                    });

                // Verify results match
                let sample_node_id = model.nodes.len() / 2;
                let cpu_uz = cpu_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let mf_uz = mf_result.displacements.iter()
                    .find(|d| d.node_id == sample_node_id)
                    .unwrap()
                    .translation.z;
                let diff = (cpu_uz - mf_uz).abs();
                println!("  Result diff:    {:.3e} m ({:.3}% error)", diff, (diff / cpu_uz.abs()) * 100.0);

                assert!(diff < 1e-2, "Matrix-free results must match CPU within tolerance");
            }
            Err(e) => {
                println!("GPU Matrix-Free:  Not available ({})", e);
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "gpu")))]
    {
        println!("GPU Solvers:      Skipped (requires macOS + 'gpu' feature)");
    }

    println!("\n{:=<80}", "");
}
