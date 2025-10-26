# Contributing to FE Engine

Thank you for your interest in contributing to FE Engine! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/fe-engine.git`
3. Add upstream remote: `git remote add upstream https://github.com/krank56/fe-engine.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Rust 1.70 or higher
- Cargo
- (Optional) Metal development tools for GPU features on macOS

### Building

```bash
cargo build
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_beam_analysis

# Run with release optimizations
cargo test --release
```

### Code Style

We follow the standard Rust style guidelines:

```bash
# Format code
cargo fmt

# Run clippy linter
cargo clippy -- -D warnings
```

Configuration files:
- `rustfmt.toml` - Code formatting rules
- `clippy.toml` - Linting rules

## Making Changes

### Branch Naming

- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Performance: `perf/description`

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for shell elements

- Implement shell element stiffness matrix
- Add local-to-global transformation
- Include tests for shell element
```

### Testing

- Add tests for all new functionality
- Ensure existing tests pass
- Include integration tests for major features
- Add benchmark tests for performance-critical code

### Documentation

- Add doc comments to all public APIs
- Update README.md for user-facing changes
- Add examples for new features
- Update CHANGELOG.md

## Pull Request Process

1. Update your branch with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Ensure all tests pass and code is formatted:
   ```bash
   cargo test
   cargo fmt
   cargo clippy
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

5. PR should include:
   - Clear description of changes
   - Link to related issues
   - Test coverage
   - Documentation updates

### PR Review Process

- Maintainers will review your PR
- Address feedback and update PR
- Once approved, maintainer will merge

## Areas for Contribution

### High Priority

- Additional element types (shell, solid, truss)
- Nonlinear analysis support
- Dynamic analysis capabilities
- Performance optimizations
- Cross-platform GPU support

### Documentation

- API documentation improvements
- Tutorial examples
- Wiki content
- Validation examples

### Testing

- Additional test cases
- Benchmarking suite
- Validation against commercial FEA tools

## Project Structure

```
fe-engine/
├── src/
│   ├── structure/      # Core model definitions
│   ├── builder/        # Model builder API
│   ├── validation/     # Validation system
│   ├── analysis/       # FEA solvers
│   ├── audit/          # Audit trail
│   └── export/         # Result export
├── tests/
│   ├── integration/    # Integration tests
│   ├── contract/       # Contract tests
│   └── benchmarks/     # Benchmark tests
└── examples/           # Usage examples
```

## Coding Guidelines

### Rust Idioms

- Use `Result<T, E>` for error handling
- Prefer iterator chains over loops
- Use meaningful variable names
- Keep functions small and focused
- Document public APIs with examples

### Performance

- Profile before optimizing
- Use benchmarks to verify improvements
- Consider memory layout for cache efficiency
- Avoid unnecessary allocations

### Error Handling

- Use `thiserror` for error types
- Provide descriptive error messages
- Include context in error chains

### Testing

- Unit tests in same file as implementation
- Integration tests in `tests/` directory
- Use `approx` crate for floating-point comparisons
- Test edge cases and error conditions

## Questions?

- Open an issue for bugs or feature requests
- Tag issues with appropriate labels
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in the project documentation and release notes.
