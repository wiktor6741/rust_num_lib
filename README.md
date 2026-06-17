# rust_num_lib

A scientific computing library written in Rust, with Python bindings via [PyO3](https://pyo3.rs/).

The goal of this project is to provide fast, low-level implementations of numerical algorithms (linear algebra, statistics) with a convenient Python interface — combining Rust's performance with Python's ergonomics.

## Modules

- **`rust_num_lib.linalg`** — linear algebra: matrix multiplication, factorizations (LU, QR, Cholesky), solving linear systems, eigenvalues and eigenvectors.
- **`rust_num_lib.stats`** — statistics: mean, standard deviation, sampling from distributions, data generation.

## Requirements

- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain, 2024 edition — Rust 1.85+)
- Python 3.8+
- [maturin](https://www.maturin.rs/) — tool for building Python extensions from Rust

## Installation and Setup

The library is not (yet) published on PyPI, so it has to be built locally from source using `maturin`.

### 1. Clone the repository

```bash
git clone <repository-url>
cd rust_num_lib
```

### 2. Create and activate a virtual environment

Using a clean `venv` is recommended (not conda — mixing the two causes conflicts during the build):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

> **Note:** if you have a conda environment active, deactivate it first (`conda deactivate`), because `maturin` will refuse to build when both `VIRTUAL_ENV` and `CONDA_PREFIX` are set.

### 3. Install maturin

```bash
pip install maturin
```

### 4. Build and install the library

```bash
maturin develop --release
```

The `--release` flag matters — without it the code compiles in debug mode (no optimizations), which is many times slower and unsuitable for benchmarking.

This command builds the extension and installs it into the active virtual environment.

### 5. Verify it works

```python
import rust_num_lib.linalg as linalg

a = linalg.Matrix.identity(3)
print(a)
```

## Tests

Unit tests (pure Rust, without PyO3) are run via Cargo:

```bash
cargo test
```

## Project Structure

```
rust_num_lib/
├── src/
│   ├── lib.rs          # entry point, PyO3 module registration
│   ├── python.rs       # Python bindings for Matrix methods
│   ├── linalg/         # linear algebra module
│   └── stats/          # statistics module
├── Cargo.toml
├── pyproject.toml
└── README.md
```