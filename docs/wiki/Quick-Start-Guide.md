# Quick Start Guide

This guide will help you get started with FE Engine in just a few minutes.

## Installation

Add FE Engine to your `Cargo.toml`:

```toml
[dependencies]
fe-engine = "0.1"
```

## Your First Analysis

Let's analyze a simple simply-supported beam with a uniform load.

### 1. Create a New Project

```bash
cargo new my-beam-analysis
cd my-beam-analysis
```

### 2. Add Dependency

Edit `Cargo.toml` and add:

```toml
[dependencies]
fe-engine = "0.1"
```

### 3. Write the Code

Edit `src/main.rs`:

```rust
use fe_engine::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create a model builder
    let mut builder = ModelBuilder::new("Simple Beam");

    // Step 2: Define material (Steel S275)
    let steel = Material {
        id: 0,
        name: "Steel S275".to_string(),
        material_type: MaterialType::Steel {
            grade: SteelGrade {
                standard: SteelStandard::Eurocode3 {
                    grade: "S275".to_string(),
                },
                yield_strength: 275e6,
                ultimate_strength: 430e6,
            },
        },
        elastic_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        thermal_expansion: 12e-6,
        code_reference: None,
    };
    builder.add_material(steel);

    // Step 3: Define section (300x200 mm UB)
    let section = Section {
        area: 0.01,           // 10,000 mm² = 0.01 m²
        inertia_y: 8.33e-6,   // Second moment of area
        inertia_z: 8.33e-6,
        torsion_constant: 1.0e-6,
    };

    // Step 4: Create geometry (6m beam with 10 elements)
    let length = 6.0;  // meters
    let num_elements = 10;
    
    let node_ids: Vec<_> = (0..=num_elements)
        .map(|i| {
            let x = (i as f64) * length / (num_elements as f64);
            builder.add_node(Point3D { x, y: 0.0, z: 0.0 })
        })
        .collect();

    // Step 5: Add beam elements
    for i in 0..num_elements {
        builder.add_beam_element(
            node_ids[i],
            node_ids[i + 1],
            0,  // material id
            section.clone()
        )?;
    }

    // Step 6: Add supports
    builder.add_support(node_ids[0], SupportType::Pinned)?;
    builder.add_support(
        node_ids[num_elements],
        SupportType::Roller {
            free_direction: Direction::X,
        }
    )?;

    // Step 7: Add load case (10 kN/m downward)
    builder
        .create_load_case("Dead Load", LoadType::Dead)
        .add_uniform_load_on_all_elements(
            10_000.0,  // 10 kN/m
            Vector3D { x: 0.0, y: 0.0, z: -1.0 },
        )?
        .finish();

    // Step 8: Build the model
    let model = builder.build()?;
    println!("✓ Model created: {} nodes, {} elements", 
             model.nodes.len(), model.elements.len());

    // Step 9: Run analysis with CPU Cholesky solver (recommended)
    let solver = CpuCholesky;  // ✅ Always use this for production
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, &model.load_cases[0])?;

    println!("✓ Analysis complete in {:.2} ms", 
             result.solver_info.solve_time.as_millis());

    // Step 10: Extract results
    let max_disp = result.max_displacement();
    let (node_id, location) = result.max_displacement_location();
    
    println!("\n=== Results ===");
    println!("Max displacement: {:.2} mm at node {}", 
             max_disp * 1000.0, node_id);
    println!("Location: x={:.2}m", location.x);

    // Check deflection limit (L/250 for Eurocode)
    if result.check_deflection_limit(250.0, length) {
        println!("✓ Deflection OK (limit = {:.2} mm)", 
                 length * 1000.0 / 250.0);
    } else {
        println!("✗ Deflection exceeds limit!");
    }

    // Get reaction forces
    let (total_force, total_moment) = result.total_reactions();
    println!("\nTotal reactions:");
    println!("  Vertical force: {:.1} kN", total_force.z.abs() / 1000.0);

    Ok(())
}
```

### 4. Run the Analysis

```bash
cargo run
```

You should see output like:

```
✓ Model created: 11 nodes, 10 elements
✓ Analysis complete in 2.34 ms

=== Results ===
Max displacement: 15.23 mm at node 5
Location: x=3.00m
✓ Deflection OK (limit = 24.00 mm)

Total reactions:
  Vertical force: 60.0 kN
```

## Understanding the Code

### Solver Selection

**Always use `CpuCholesky` for production work:**

```rust
let solver = CpuCholesky;  // ✅ Stable, fast, reliable
```

The `CpuCholesky` solver is:
- **Recommended for all production use**
- Direct solver providing exact solutions (within numerical precision)
- Efficient for models up to ~50,000 DOF
- Well-tested and stable

**Do not use GPU solver** - it's experimental and currently 8-10× slower than CPU. See the main [README](../../README.md#solvers) for detailed comparison.

### Model Builder Pattern

FE Engine uses a fluent builder pattern:

```rust
let mut builder = ModelBuilder::new("Model Name");
builder.add_material(...);
builder.add_node(...);
let model = builder.build()?;
```

### Units

FE Engine uses SI base units internally:
- Length: meters (m)
- Force: Newtons (N)
- Stress: Pascals (Pa = N/m²)
- Density: kg/m³

Convert your inputs accordingly:
- 1 kN = 1,000 N
- 1 mm = 0.001 m
- 1 MPa = 1,000,000 Pa

### Error Handling

All operations that can fail return `Result<T, E>`:

```rust
// Handle errors explicitly
match builder.add_beam_element(...) {
    Ok(element_id) => println!("Element {} created", element_id),
    Err(e) => eprintln!("Error: {}", e),
}

// Or use ? operator to propagate
builder.add_beam_element(...)?;
```

## Next Steps

- Learn about [Materials and Sections](Materials-and-Sections.md)
- Explore different [Element Types](Elements.md)
- Read about [Load Cases](Loads-and-Supports.md)
- Try more [Examples](../examples/)

## Getting Help

- Check the [API Documentation](https://docs.rs/fe-engine)
- Browse [Examples](../examples/)
- Open an [Issue](https://github.com/krank56/fe-engine/issues) on GitHub
