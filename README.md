# rust_num_lib

A small hobby scientific computation library written in Rust, with Python bindings via [PyO3](https://pyo3.rs/). It covers some basic linear algebra and statistics, and was built mainly for fun and learning.

> **Disclaimer:** this is a hobby project. There is **no guarantee of correctness or numerical stability**, and the algorithms have not been validated against established libraries beyond basic checks. It is **not intended for serious research, production use, or anything where wrong results would matter**. For real work, use a well-tested library such as [NumPy](https://numpy.org/) or [SciPy](https://scipy.org/).

## Modules

- **`rust_num_lib.linalg`**: linear algebra: matrix multiplication, factorizations (LU, QR, Cholesky), solving linear systems, eigenvalues and eigenvectors.
- **`rust_num_lib.stats`**: statistics: mean, median, standard deviation, variance, and sampling from distributions.

## Installation

The recommended way is to install from [PyPI](https://pypi.org/project/rust-num-lib/):

```bash
pip install rust-num-lib
```

Prebuilt wheels are published for Linux, Windows and macOS (Intel and Apple Silicon), so you don't need Rust installed. Python 3.9 or newer is required.

Note that the package is installed as `rust-num-lib` but imported as `rust_num_lib`.

### Quick start

```python
from rust_num_lib import linalg, stats

a = linalg.matrix([[2, 1], [1, 3]])
b = linalg.vector([1, 2])

x = linalg.solve(a, b)       # solve A x = b
print(x)

print(a.determinant())
print(stats.mean(b))
```

The package ships type stubs with docstrings, so editors such as VS Code show documentation when you hover over a function. A full reference is also available as a [PDF](https://github.com/wiktor6741/rust_num_lib/blob/main/docs/rust_num_lib.pdf).

## Building from source

Only needed if you want to modify the library or your platform has no prebuilt wheel.

### Requirements

- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain, 2024 edition, Rust 1.85+)
- Python 3.9+
- [maturin](https://www.maturin.rs/), the tool that builds Python extensions from Rust

### 1. Clone the repository

```bash
git clone https://github.com/wiktor6741/rust_num_lib.git
cd rust_num_lib
```

### 2. Create and activate a virtual environment

Using a clean `venv` is recommended (not conda, since mixing the two causes conflicts during the build):

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

The `--release` flag matters. Without it the code compiles in debug mode (no optimizations), which is many times slower.

This builds the extension and installs it into the active virtual environment in editable mode.

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
├── python/rust_num_lib/  # Python package files: type stubs (.pyi) and __init__.py
├── docs/               # PDF reference and the script that generates it
├── .github/workflows/  # CI: builds wheels and publishes releases
├── Cargo.toml
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

[MIT](https://github.com/wiktor6741/rust_num_lib/blob/main/LICENSE)
