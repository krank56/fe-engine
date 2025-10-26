# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-26

### Added

- Initial release of FE Engine
- Core structural model with nodes, elements, materials, and loads
- Model builder API for fluent model construction
- Comprehensive validation system
- **CPU Cholesky solver** (stable, production-ready)
- **Experimental GPU solver** for macOS Metal (not recommended for production)
- Beam and Frame2D element types
- Support for multiple material standards (Eurocode, AISC, BS, ACI)
- Multiple load case support (dead, live, wind, seismic, thermal)
- Audit trail system for traceability
- CSV export for analysis results
- Comprehensive test suite including integration and benchmark tests
- Example programs demonstrating usage

### Features

#### Structural Elements
- 3D beam element with 6 DOF per node
- 2D frame element with 3 DOF per node

#### Materials
- Steel materials with Eurocode 3, AISC, BS 5950 standards
- Concrete materials with Eurocode 2, ACI 318, BS 8110 standards

#### Analysis
- Linear static analysis
- **CPU Cholesky solver** (stable, production-ready, recommended for all use)
- **Experimental GPU solver** for macOS Metal (not recommended - 8-10× slower than CPU)
- Deflection limit checking
- Reaction force calculation

#### Export
- CSV export for displacements and reactions
- Audit trail export

[Unreleased]: https://github.com/krank56/fe-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/krank56/fe-engine/releases/tag/v0.1.0
