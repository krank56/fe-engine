# FE Engine

A high-performance finite element analysis (FEA) engine for structural engineering, built in Rust.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **High Performance CPU Solver**: Sparse Cholesky decomposition for efficient direct solving (recommended for production)
- **Structural Elements**: Support for beam and 2D frame elements
- **Material Library**: Built-in support for steel and concrete materials with international standards (Eurocode, AISC, BS)
- **Load Cases**: Multiple load case support (dead, live, wind, seismic, thermal)
- **Validation**: Comprehensive validation system for model integrity
- **Audit Trail**: Full traceability of analysis operations
- **Export**: CSV export for results and integration with other tools
- **GPU Solver (Experimental)**: Optional Metal-based GPU solver for macOS (not recommended for production use)

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fe-engine = "0.1"
```

### Basic Example

```rust
use fe_engine::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a model builder
    let mut builder = ModelBuilder::new("Simple Beam");

    // Define material
    let mat = Material {
        id: 0,
        name: "Steel S355".to_string(),
        material_type: MaterialType::Steel {
            grade: SteelGrade {
                standard: SteelStandard::Eurocode3 {
                    grade: "S355".to_string(),
                },
                yield_strength: 355e6,
                ultimate_strength: 490e6,
            },
        },
        elastic_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        thermal_expansion: 12e-6,
        code_reference: None,
    };
    builder.add_material(mat);

    // Define section properties
    let section = Section {
        area: 0.01,
        inertia_y: 8.33e-6,
        inertia_z: 8.33e-6,
        torsion_constant: 1.0e-6,
    };

    // Create nodes
    let n1 = builder.add_node(Point3D { x: 0.0, y: 0.0, z: 0.0 });
    let n2 = builder.add_node(Point3D { x: 5.0, y: 0.0, z: 0.0 });

    // Add element
    builder.add_beam_element(n1, n2, 0, section)?;

    // Add supports
    builder.add_support(n1, SupportType::Fixed)?;
    builder.add_support(n2, SupportType::Roller {
        free_direction: Direction::X,
    })?;

    // Add load case
    builder
        .create_load_case("Dead Load", LoadType::Dead)
        .add_uniform_load_on_all_elements(
            10_000.0,
            Vector3D { x: 0.0, y: 0.0, z: -1.0 },
        )?
        .finish();

    // Build model
    let model = builder.build()?;

    // Run analysis
    let solver = CpuCholesky;
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, &model.load_cases[0])?;

    // Check results
    let max_disp = result.max_displacement();
    println!("Max displacement: {:.3} mm", max_disp * 1000.0);

    Ok(())
}
```

## Architecture

The engine is organized into several modules:

### Core Modules

- **`structure`**: Core structural model definitions (nodes, elements, materials, loads)
- **`builder`**: Fluent API for building structural models
- **`validation`**: Model validation and error checking
- **`analysis`**: FEA solver implementations (CPU and GPU)
- **`audit`**: Audit trail for traceability
- **`export`**: Result export functionality

### Analysis Pipeline

```
Model Builder → Validation → Assembly → Solver → Results
                                ↓
                          Audit Trail
```

## Supported Elements

- **Beam Element**: 3D beam element with 6 DOF per node (3 translations + 3 rotations)
- **Frame2D Element**: 2D frame element with 3 DOF per node (2 translations + 1 rotation)

## Material Standards

### Steel
- Eurocode 3 (EN 1993)
- AISC (American Institute of Steel Construction)
- BS 5950 (British Standard)

### Concrete
- Eurocode 2 (EN 1992)
- ACI 318 (American Concrete Institute)
- BS 8110 (British Standard)

## Load Types

- Dead Load
- Live Load
- Wind Load
- Seismic Load
- Thermal Load
- Construction Load
- Prestress

## Solvers

### CPU Cholesky Solver (Stable - Recommended)

The default and **recommended** solver for production use. Uses sparse Cholesky decomposition via `nalgebra-sparse`.

**Characteristics:**
- **Direct solver**: Provides exact solution (within numerical precision)
- **Production-ready**: Stable, well-tested, and reliable
- **Efficient**: Optimal for models up to ~50,000 DOF
- **Backed by established research**: Based on proven sparse direct methods

**When to use:**
- ✅ All production environments
- ✅ Models requiring guaranteed accuracy
- ✅ Small to medium-sized models (<50K DOF)
- ✅ When deterministic results are needed

**References:**
- Golub & Van Loan, "Matrix Computations" (2013)
- Davis, "Direct Methods for Sparse Linear Systems" (SIAM, 2006)
- [nalgebra-sparse documentation](https://docs.rs/nalgebra-sparse/)

**Benchmark Results** (10,000 DOF portal frame):
- Solution time: ~250ms
- Memory usage: ~180MB
- Convergence: Single-step (direct method)

### GPU Metal Solver (Experimental - Not Recommended)

An experimental GPU-accelerated iterative solver using Metal Compute on macOS.

**⚠️ Current Limitations (DO NOT use in production):**
- 8-10× **slower** than CPU solver for typical models
- Excessive kernel dispatches (~8-10 per iteration)
- Inefficient atomic operations causing GPU stalls
- macOS-only, limited platform support
- Minimal testing and validation
- No convergence guarantees for ill-conditioned systems

**Known Issues:**
1. **Performance regression**: GPU overhead exceeds computational benefits for models <100K DOF
2. **Synchronization bottlenecks**: CPU-GPU data transfer dominates runtime
3. **Iterative instability**: PCG may fail to converge for poorly-conditioned matrices

**Status:** This solver exists as a research prototype and requires significant optimization before production use.

**When to consider (future versions only):**
- Models exceeding 100,000 DOF (not yet validated)
- After major performance optimizations are implemented
- Non-critical applications where approximate solutions are acceptable

**For now, always use `CpuCholesky` solver.**

## GPU Acceleration (Experimental)

⚠️ **Not recommended for production use.** The GPU solver is currently 8-10× slower than the CPU solver and has known performance issues.

Enable GPU features (macOS only) for research/testing purposes:

```toml
[dependencies]
fe-engine = { version = "0.1", features = ["gpu"] }
```

**Always use the `CpuCholesky` solver for production work.** See the [Solvers](#solvers) section for detailed comparison.

## Performance

The engine is optimized for performance:

- **Sparse matrix operations** using `nalgebra-sparse` with efficient Cholesky decomposition
- **Efficient memory layout** for cache performance
- **Production-ready CPU solver** handles models up to ~50K DOF efficiently
- **Experimental GPU solver** (not recommended - see [Solvers](#solvers) section)
- Parallel assembly (planned)

## Development

### Building

```bash
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run benchmarks
cargo test --test simple_beam --release
cargo test --test portal_frame --release
```

### Examples

```bash
# Quick start example
cargo run --example quickstart

# Large benchmark
cargo run --example benchmark_large --release
```

## Documentation

Generate and view documentation:

```bash
cargo doc --open
```

For detailed documentation, see the [Wiki](https://github.com/krank56/fe-engine/wiki).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Additional element types (shell, solid)
- [ ] Nonlinear analysis
- [ ] Dynamic analysis
- [ ] Parallel assembly
- [ ] Cross-platform GPU support (Vulkan/CUDA)
- [ ] Python bindings
- [ ] Web assembly support

## Author

[@krank56](https://github.com/krank56)

## Acknowledgments

Built with:
- [nalgebra](https://nalgebra.org/) - Linear algebra library
- [metal-rs](https://github.com/gfx-rs/metal-rs) - Metal API bindings for GPU acceleration
