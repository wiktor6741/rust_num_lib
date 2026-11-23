// src/python.rs
//
// Python bindings for the `Matrix` class (methods exposed via PyO3).
// Free functions live in lib.rs; this file holds everything that is
// a method or operator on a single matrix.

use pyo3::prelude::*;
use crate::linalg::{self, Matrix, matmul};

#[pymethods]
impl Matrix {
    /// Creates a new zero-filled matrix of the given size.
    ///
    /// ```python
    /// m = linalg.Matrix(3, 4)  # 3x4 matrix of zeros
    /// ```
    #[new]
    fn py_new(rows: usize, cols: usize) -> Self {
        Matrix::zeros(rows, cols)
    }

    // ---- static constructors ----

    /// Creates a `rows x cols` matrix filled with zeros.
    #[staticmethod]
    #[pyo3(name = "zeros")]
    fn py_zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::zeros(rows, cols)
    }

    /// Creates a `size x size` identity matrix.
    #[staticmethod]
    #[pyo3(name = "identity")]
    fn py_identity(size: usize) -> Matrix {
        Matrix::identity(size)
    }

    /// Creates a `rows x cols` matrix with values drawn uniformly from `[min, max)`.
    #[staticmethod]
    #[pyo3(name = "random")]
    fn py_random(rows: usize, cols: usize, min: f64, max: f64) -> Matrix {
        Matrix::random(rows, cols, min, max)
    }

    /// Creates a matrix from a flat list of values with explicit dimensions.
    ///
    /// The list is interpreted in row-major order and must have exactly
    /// `rows * cols` elements.
    ///
    /// ```python
    /// m = linalg.Matrix.from_data([1, 2, 3, 4], 2, 2)
    /// ```
    #[staticmethod]
    #[pyo3(name = "from_data")]
    fn py_from_data(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        Matrix::from_data(data, rows, cols)
    }

    // ---- representation ----

    fn __repr__(&self) -> String {
        format!("Matrix({}x{})", self.rows, self.cols)
    }

    fn __str__(&self) -> String {
        format!("{}", self)
    }

    // ---- properties ----

    /// Number of rows.
    #[getter]
    fn rows(&self) -> usize { self.rows }

    /// Number of columns.
    #[getter]
    fn cols(&self) -> usize { self.cols }

    // ---- operators ----

    /// Matrix multiplication via the `@` operator.
    ///
    /// ```python
    /// c = a @ b
    /// ```
    fn __matmul__(&self, other: &Matrix) -> PyResult<Matrix> {
        Ok(matmul(self, other)?)
    }

    /// Element-wise matrix addition via the `+` operator.
    fn __add__(&self, other: &Matrix) -> PyResult<Matrix> {
        Ok((self + other)?)
    }

    /// Scalar multiplication via the `*` operator (`matrix * scalar`).
    fn __mul__(&self, scalar: f64) -> Matrix {
        self * scalar
    }

    /// Scalar multiplication with the scalar on the left (`scalar * matrix`).
    fn __rmul__(&self, scalar: f64) -> Matrix {
        self * scalar
    }

    // ---- indexing ----

    /// Element access via `m[i, j]`.
    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        let (r, c) = idx;
        if r >= self.rows || c >= self.cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
        }
        Ok(self[(r, c)])
    }

    /// Element assignment via `m[i, j] = value`.
    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        let (r, c) = idx;
        if r >= self.rows || c >= self.cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
        }
        self[(r, c)] = value;
        Ok(())
    }

    // ---- methods (matrix properties) ----

    /// Returns the transpose as a new matrix (does not modify in place).
    #[pyo3(name = "transpose")]
    fn py_transpose(&self) -> Matrix {
        self.transposed()
    }

    /// Determinant of a square matrix (computed via LU factorization).
    #[pyo3(name = "determinant")]
    fn py_determinant(&self) -> PyResult<f64> {
        Ok(self.determinant()?)
    }

    /// Trace — the sum of the diagonal elements.
    #[pyo3(name = "trace")]
    fn py_trace(&self) -> f64 {
        self.trace()
    }

    /// Frobenius norm — the square root of the sum of squared elements.
    #[pyo3(name = "norm")]
    fn py_norm(&self) -> f64 {
        self.norm()
    }

    /// Rank of the matrix (number of non-zero pivots from LU factorization).
    #[pyo3(name = "rank")]
    fn py_rank(&self) -> usize {
        self.rank()
    }

    /// Eigenvalues, approximated with the QR algorithm.
    ///
    /// `steps` controls the number of QR iterations (defaults to a built-in value).
    ///
    /// ```python
    /// vals = a.eigenvalues()
    /// vals = a.eigenvalues(200)
    /// ```
    #[pyo3(name = "eigenvalues")]
    fn py_eigenvalues(&self, steps: Option<u32>) -> PyResult<Vec<f64>> {
        Ok(self.eigenvalues(steps)?)
    }

    /// Eigenvalues and eigenvectors, returned as `(values, vectors)`.
    ///
    /// For symmetric matrices the eigenvectors are the columns of the
    /// accumulated orthogonal matrix; otherwise they are recovered from
    /// the Schur form.
    #[pyo3(name = "eigenvectors")]
    fn py_eigenv_full(&self, steps: Option<u32>) -> PyResult<(Vec<f64>, Vec<Matrix>)> {
        Ok(self.eigenv_full(steps)?)
    }

    /// Returns row `i` as a `1 x cols` matrix.
    #[pyo3(name = "row")]
    fn py_row(&self, i: usize) -> PyResult<Matrix> {
        Ok(self.row_matrix(i)?)
    }

    /// Returns column `i` as a `rows x 1` matrix.
    #[pyo3(name = "col")]
    fn py_col(&self, i: usize) -> PyResult<Matrix> {
        Ok(self.col_matrix(i)?)
    }

    /// Extracts a sub-matrix with optional ranges and steps along each axis.
    ///
    /// Negative steps reverse the corresponding axis. Each range is a
    /// `(start, end)` tuple (end-exclusive); `None` selects the full axis.
    ///
    /// ```python
    /// sub = m.slice((0, 3), None, (1, 4), None)   # rows 0..3, cols 1..4
    /// rev = m.slice(None, -1, None, None)          # reverse all rows
    /// ```
    #[pyo3(name = "slice")]
    fn py_slice(
        &self,
        row_range: Option<(usize, usize)>,
        row_step: Option<i32>,
        col_range: Option<(usize, usize)>,
        col_step: Option<i32>,
    ) -> PyResult<Matrix> {
        Ok(self.slice(row_range, row_step, col_range, col_step)?)
    }

    /// Applies a Python function along an axis, reducing each row or column
    /// to a single value.
    ///
    /// `axis = 0` applies `f` to each column (result is `1 x cols`),
    /// `axis = 1` applies `f` to each row (result is `rows x 1`).
    /// The callable `f` receives a `Matrix` (the row/column) and returns a float.
    ///
    /// ```python
    /// col_means = m.apply_along_axis(stats.mean, 0)
    /// ```
    #[pyo3(name = "apply_along_axis")]
    fn py_apply_along_axis(&self, f: Py<PyAny>, axis: u32, py: Python) -> PyResult<Matrix> {
        Ok(linalg::apply_along_axis(self, |v| {
            f.call1(py, (v.clone(),))
                .unwrap()
                .extract::<f64>(py)
                .unwrap()
        }, axis)?)
    }
}