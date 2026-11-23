//! # rust_num_lib
//!
//! A scientific computing library written in Rust with Python bindings via PyO3.
//!
//! The library exposes two submodules to Python:
//! - [`linalg`] — linear algebra (matrix multiplication, factorizations, eigenvalues, solving systems)
//! - [`stats`]  — statistics (descriptive statistics, sampling from distributions)
//!
//! From Python:
//! ```python
//! import rust_num_lib.linalg as linalg
//! import rust_num_lib.stats as stats
//!
//! a = linalg.matrix([[1, 2], [3, 4]])
//! vals = a.eigenvalues()
//! ```

pub mod linalg;
pub mod stats;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
mod py_bindings {
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;
    use crate::linalg::{self, Matrix, MatrixError};
    use crate::stats;

    /// Converts a library error into a Python `ValueError`.
    impl From<MatrixError> for PyErr {
        fn from(e: MatrixError) -> PyErr {
            PyValueError::new_err(e.to_string())
        }
    }

    /// Top-level Python module `rust_num_lib`.
    ///
    /// Registers the `linalg` and `stats` submodules so that
    /// `import rust_num_lib.linalg` and `import rust_num_lib.stats` work.
    #[pymodule]
    #[pyo3(name = "rust_num_lib")]
    fn rust_num_lib(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        let linalg_mod = PyModule::new(py, "linalg")?;
        register_linalg(&linalg_mod)?;
        m.add_submodule(&linalg_mod)?;

        let stats_mod = PyModule::new(py, "stats")?;
        register_stats(&stats_mod)?;
        m.add_submodule(&stats_mod)?;

        // Register submodules in sys.modules so that
        // `import rust_num_lib.linalg` / `.stats` resolve correctly.
        let sys_modules = py.import("sys")?.getattr("modules")?;
        sys_modules.set_item("rust_num_lib.linalg", &linalg_mod)?;
        sys_modules.set_item("rust_num_lib.stats", &stats_mod)?;

        Ok(())
    }

    /// Registers all linear algebra functions and the `Matrix` class
    /// on the `linalg` submodule.
    fn register_linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Matrix>()?;
        m.add_function(wrap_pyfunction!(py_matmul_chained, m)?)?;
        m.add_function(wrap_pyfunction!(py_solve, m)?)?;
        m.add_function(wrap_pyfunction!(py_solve_multiple, m)?)?;
        m.add_function(wrap_pyfunction!(py_qr_factorize, m)?)?;
        m.add_function(wrap_pyfunction!(py_lu_factorize, m)?)?;
        m.add_function(wrap_pyfunction!(py_cholesky, m)?)?;
        m.add_function(wrap_pyfunction!(py_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_vector, m)?)?;
        Ok(())
    }

    /// Builds a matrix from a 2D list, inferring dimensions from the shape.
    ///
    /// All rows must have the same length.
    ///
    /// ```python
    /// a = linalg.matrix([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
    /// ```
    #[pyfunction]
    #[pyo3(name = "matrix")]
    fn py_matrix(data: Vec<Vec<f64>>) -> PyResult<Matrix> {
        let rows = data.len();
        if rows == 0 {
            return Err(PyValueError::new_err("empty matrix"));
        }
        let cols = data[0].len();

        for row in &data {
            if row.len() != cols {
                return Err(PyValueError::new_err("rows have different lengths"));
            }
        }

        let flat: Vec<f64> = data.into_iter().flatten().collect();

        Ok(Matrix::from_data(flat, rows, cols))
    }

    /// Builds a column vector (n x 1 matrix) from a flat list.
    ///
    /// ```python
    /// v = linalg.vector([1, 2, 3])  # 3x1 column vector
    /// ```
    #[pyfunction]
    #[pyo3(name = "vector")]
    fn py_vector(data: Vec<f64>) -> PyResult<Matrix> {
        Ok(linalg::vector(data))
    }

    /// Multiplies a chain of matrices using an optimal multiplication order.
    ///
    /// Equivalent to `A @ B @ C @ ...` but chooses the parenthesization
    /// that minimizes the total number of scalar multiplications.
    ///
    /// ```python
    /// result = linalg.matmul_chained([a, b, c, d])
    /// ```
    #[pyfunction]
    #[pyo3(name = "matmul_chained")]
    fn py_matmul_chained(matrices: Vec<Matrix>) -> PyResult<Matrix> {
        Ok(linalg::matmul_chained(&matrices)?)
    }

    /// Solves the linear system `A x = b` for `x`.
    ///
    /// ```python
    /// x = linalg.solve(a, b)
    /// ```
    #[pyfunction]
    #[pyo3(name = "solve")]
    fn py_solve(a: &Matrix, b: &Matrix) -> PyResult<Matrix> {
        Ok(linalg::gauss_elimination(a, b)?)
    }

    /// Solves `A x = b` for multiple right-hand sides, factorizing `A` once.
    ///
    /// More efficient than calling `solve` repeatedly with the same `A`.
    ///
    /// ```python
    /// xs = linalg.solve_multiple(a, [b1, b2, b3])
    /// ```
    #[pyfunction]
    #[pyo3(name = "solve_multiple")]
    fn py_solve_multiple(a: &Matrix, bs: Vec<Matrix>) -> PyResult<Vec<Matrix>> {
        Ok(linalg::solve_multiple(a, &bs)?)
    }

    /// Computes the QR factorization of a matrix, returning `(Q, R)`.
    ///
    /// `Q` has orthonormal columns and `R` is upper triangular, with `A = Q R`.
    #[pyfunction]
    #[pyo3(name = "qr_factorize")]
    fn py_qr_factorize(a: &Matrix) -> PyResult<(Matrix, Matrix)> {
        Ok(linalg::qr_factorize(a)?)
    }

    /// Computes the LU factorization with partial pivoting.
    ///
    /// Returns `(L, U, perm)` such that `P A = L U`, where `perm` is the
    /// row permutation vector.
    #[pyfunction]
    #[pyo3(name = "lu_factorize")]
    fn py_lu_factorize(a: &Matrix) -> PyResult<(Matrix, Matrix, Vec<usize>)> {
        Ok(linalg::doolitle_factorize(a)?)
    }

    /// Computes the Cholesky factorization of a symmetric positive-definite matrix.
    ///
    /// Returns the lower-triangular `L` such that `A = L Lᵀ`.
    /// Fails if the matrix is not positive definite.
    #[pyfunction]
    #[pyo3(name = "cholesky")]
    fn py_cholesky(a: &Matrix) -> PyResult<Matrix> {
        Ok(linalg::cholesky(a)?)
    }

    /// Registers all statistics functions on the `stats` submodule.
    fn register_stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_mean, m)?)?;
        m.add_function(wrap_pyfunction!(py_median, m)?)?;
        m.add_function(wrap_pyfunction!(py_std_dev, m)?)?;
        m.add_function(wrap_pyfunction!(py_variance, m)?)?;
        m.add_function(wrap_pyfunction!(py_normal_sample, m)?)?;
        m.add_function(wrap_pyfunction!(py_multivariate_normal_sample, m)?)?;
        Ok(())
    }

    /// Arithmetic mean of all elements in the matrix.
    #[pyfunction]
    #[pyo3(name = "mean")]
    fn py_mean(m: &Matrix) -> PyResult<f64> {
        Ok(stats::mean(m))
    }

    /// Median of all elements in the matrix.
    #[pyfunction]
    #[pyo3(name = "median")]
    fn py_median(m: &Matrix) -> PyResult<f64> {
        Ok(stats::median(m))
    }

    /// Standard deviation of all elements in the matrix.
    #[pyfunction]
    #[pyo3(name = "std_dev")]
    fn py_std_dev(m: &Matrix) -> PyResult<f64> {
        Ok(stats::std_dev(m))
    }

    /// Variance of all elements in the matrix.
    #[pyfunction]
    #[pyo3(name = "variance")]
    fn py_variance(m: &Matrix) -> PyResult<f64> {
        Ok(stats::variance(m))
    }

    /// Draws `n` samples from a normal distribution `N(mu, sigma^2)`.
    ///
    /// Returns an `n x 1` column vector of samples.
    ///
    /// ```python
    /// noise = stats.normal_sample(100, 0.0, 1.0)
    /// ```
    #[pyfunction]
    #[pyo3(name = "normal_sample")]
    fn py_normal_sample(n: usize, mu: f64, sigma: f64) -> PyResult<Matrix> {
        Ok(stats::normal_sample(n, mu, sigma))
    }

    /// Draws samples from a multivariate normal distribution `N(mu, Sigma)`.
    ///
    /// `mu` is the mean vector and `Sigma` the covariance matrix (must be
    /// symmetric positive-definite). Returns an `n_samples x dim` matrix
    /// where each row is one sample.
    #[pyfunction]
    #[pyo3(name = "multivariate_normal_sample")]
    fn py_multivariate_normal_sample(
        n_samples: usize,
        mu: &Matrix,
        sigma: &Matrix,
    ) -> PyResult<Matrix> {
        Ok(stats::multivariete_normal_sample(n_samples, mu, sigma)?)
    }
}