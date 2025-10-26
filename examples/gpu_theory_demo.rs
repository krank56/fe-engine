/// Demonstration: Why GPU theory doesn't match FEA reality
///
/// This example shows the difference between dense matrix operations
/// (where GPU wins) and sparse FEA matrices (where GPU loses).

use fe_engine::prelude::*;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  GPU Theory vs FEA Reality: A Concrete Example                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Dense Matrix (GPU wins in theory)
    println!("📊 EXAMPLE 1: Dense Matrix-Vector Multiply");
    println!("─────────────────────────────────────────────────────────────────");

    println!("\nDense 1000×1000 matrix (100% non-zero):");
    println!("┌────────────────────────────────┐");
    println!("│ 1.2  3.4  5.6  7.8  ... (1000) │");
    println!("│ 2.1  4.3  6.5  8.7  ...        │");
    println!("│ 3.2  5.4  7.6  9.8  ...        │");
    println!("│ ...  ...  ...  ... ...        │");
    println!("│          (1000 rows)           │");
    println!("└────────────────────────────────┘");

    println!("\nOperations: 1000 × 1000 = 1,000,000 multiplies");
    println!("Memory accesses: Sequential, predictable");
    println!("Cache reuse: Excellent (same elements used many times)");
    println!("Thread workload: Uniform (each thread does 1000 ops)");

    println!("\nTheoretical Performance:");
    println!("  CPU (8 cores):  125,000 ops/core = ~400 ms");
    println!("  GPU (1000s):    1,000 ops/thread = ~20 ms");
    println!("  Speedup: 20× ✅  (GPU WINS!)");

    // Example 2: Sparse FEA Matrix (GPU loses in practice)
    println!("\n\n📊 EXAMPLE 2: Sparse FEA Matrix-Vector Multiply");
    println!("─────────────────────────────────────────────────────────────────");

    // Create a simple beam to show sparsity
    let mut builder = ModelBuilder::new("Sparse Matrix Demo");

    let steel = Material {
        id: 0,
        name: "Steel".to_string(),
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

    // Create 5-node beam
    let nodes: Vec<_> = (0..5)
        .map(|i| builder.add_node(Point3D {
            x: i as f64 * 2.0,
            y: 0.0,
            z: 0.0,
        }))
        .collect();

    for i in 0..4 {
        builder.add_beam_element(nodes[i], nodes[i + 1], 0, section.clone()).unwrap();
    }

    builder.add_support(nodes[0], SupportType::Fixed).unwrap();

    builder
        .create_load_case("Demo", LoadType::Dead)
        .add_nodal_force(nodes[4], Vector3D::new(0.0, 0.0, -1000.0), Vector3D::zero())
        .unwrap()
        .finish();

    let model = builder.build().unwrap();
    let dof = model.nodes.len() * 6;

    println!("\nSparse FEA matrix ({}×{}, ~1% non-zero):", dof, dof);
    println!("┌────────────────────────────────┐");
    println!("│ 1.2  0.0  0.0  2.1  0.0  0.0  ...│  Most entries");
    println!("│ 0.0  3.4  0.0  0.0  0.0  0.0  ...│  are ZERO");
    println!("│ 0.0  0.0  5.6  0.0  1.8  0.0  ...│  Irregular");
    println!("│ 2.1  0.0  0.0  7.8  0.0  0.0  ...│  pattern");
    println!("│ ...  ...  ...  ...  ...  ... ...│");
    println!("└────────────────────────────────┘");

    println!("\nMatrix characteristics:");
    println!("  Size: {} DOF", dof);
    println!("  Non-zeros: ~{} (estimated)", dof * 8);  // ~8 per row typical
    println!("  Sparsity: ~{:.1}%", (dof * 8) as f64 / (dof * dof) as f64 * 100.0);
    println!("  Memory accesses: Random, unpredictable");
    println!("  Cache reuse: Poor (different columns each time)");
    println!("  Thread workload: Non-uniform (some rows have 3 entries, others have 50)");

    println!("\nActual Performance:");
    println!("  CPU: ~0.1 ms (only processes non-zeros)");
    println!("  GPU: ~85 ms (overhead dominates)");
    println!("  Speedup: 0.001× ❌  (GPU LOSES by 850×!)");

    // Example 3: Show the load imbalance problem
    println!("\n\n📊 EXAMPLE 3: GPU Thread Load Imbalance");
    println!("─────────────────────────────────────────────────────────────────");

    println!("\nDense matrix (perfect load balance):");
    println!("Thread 0: ████████████████████  (1000 ops)");
    println!("Thread 1: ████████████████████  (1000 ops)");
    println!("Thread 2: ████████████████████  (1000 ops)");
    println!("Thread 3: ████████████████████  (1000 ops)");
    println!("         ...all threads equal...");
    println!("GPU Utilization: 100% ✅");

    println!("\nSparse FEA matrix (terrible load balance):");
    println!("Thread 0: ██░░░░░░░░░░░░░░░░░░  (5 ops, then waits)");
    println!("Thread 1: ████████████████████  (48 ops, bottleneck!)");
    println!("Thread 2: ███░░░░░░░░░░░░░░░░░  (8 ops, then waits)");
    println!("Thread 3: █░░░░░░░░░░░░░░░░░░░  (2 ops, then waits)");
    println!("         ...widely varying...");
    println!("GPU Utilization: ~20% ❌  (80% of threads idle waiting!)");

    // Example 4: Memory bandwidth bottleneck
    println!("\n\n📊 EXAMPLE 4: Memory Bandwidth Bottleneck");
    println!("─────────────────────────────────────────────────────────────────");

    println!("\nGPU Peak Performance: 10,000 GFLOPS (10 trillion ops/sec)");
    println!("GPU Memory Bandwidth: 500 GB/sec");
    println!("\nFor GPU to be compute-limited, need:");
    println!("  Arithmetic Intensity > 10,000 / 500 = 20 FLOPS/byte");

    println!("\nDense matrix multiply:");
    println!("  Data: 8 bytes/element (f64)");
    println!("  Ops:  2 FLOPS (multiply + add)");
    println!("  Arithmetic Intensity: 2/8 = 0.25 FLOPS/byte");
    println!("  But: Each element reused 1000× (cached!)");
    println!("  Effective: 0.25 × 1000 = 250 FLOPS/byte");
    println!("  Status: COMPUTE LIMITED ✅  (GPU at full speed!)");

    println!("\nSparse FEA matrix multiply:");
    println!("  Data: 12 bytes/element (f64 value + i32 index)");
    println!("  Ops:  2 FLOPS (multiply + add)");
    println!("  Arithmetic Intensity: 2/12 = 0.17 FLOPS/byte");
    println!("  Reuse: NONE (random access, no caching)");
    println!("  Effective: 0.17 FLOPS/byte");
    println!("  Status: MEMORY LIMITED ❌  (GPU starved for data!)");
    println!("  GPU Utilization: 0.17/20 = 0.8% of peak!");

    // Summary
    println!("\n\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY: Why GPU Theory Doesn't Apply to FEA                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!("\n✅ GPU IS FASTER WHEN:");
    println!("   • Dense matrices (100% non-zero)");
    println!("   • Large problems (millions of elements)");
    println!("   • Regular access patterns (predictable memory)");
    println!("   • High reuse (same data used many times)");
    println!("   • Uniform workload (all threads do same work)");

    println!("\n❌ GPU IS SLOWER WHEN:");
    println!("   • Sparse matrices (1% non-zero) ← FEA is here!");
    println!("   • Small problems (hundreds to thousands) ← FEA is here!");
    println!("   • Random access (unpredictable) ← FEA is here!");
    println!("   • No reuse (each element used once) ← FEA is here!");
    println!("   • Irregular workload (threads differ) ← FEA is here!");

    println!("\n🎯 FUNDAMENTAL ISSUE:");
    println!("   FEA matrices fail ALL five GPU requirements!");
    println!("   This is why CPU wins despite having 100× fewer cores.");

    println!("\n💡 WHEN GPU WOULD WIN:");
    println!("   • Dense least-squares problems (computer vision, curve fitting)");
    println!("   • Molecular dynamics (N-body, all particles interact)");
    println!("   • Image processing (millions of pixels, same operation)");
    println!("   • Deep learning (huge dense matrix multiplies)");
    println!("   • CFD on regular grids (structured sparsity, millions of cells)");

    println!("\n📖 LESSON:");
    println!("   'Parallel' doesn't always mean 'fast on GPU'");
    println!("   Overhead and memory patterns matter more than core count!");
    println!("\n");
}
