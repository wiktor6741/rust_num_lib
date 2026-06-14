// src/python.rs
use pyo3::prelude::*;
use crate::linalg::{Matrix, matmul};

#[pymethods]
impl Matrix {
    #[new]
    fn py_new(rows: usize, cols: usize) -> Self {
        Matrix::zeros(rows, cols)
    }

    // ---- konstruktory statyczne ----
    #[staticmethod]
    #[pyo3(name = "zeros")]
    fn py_zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::zeros(rows, cols)
    }

    #[staticmethod]
    #[pyo3(name = "identity")]
    fn py_identity(size: usize) -> Matrix {
        Matrix::identity(size)
    }

    #[staticmethod]
    #[pyo3(name = "random")]
    fn py_random(rows: usize, cols: usize, min: f64, max: f64) -> Matrix {
        Matrix::random(rows, cols, min, max)
    }

    #[staticmethod]
    #[pyo3(name = "from_data")]
    fn py_from_data(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        Matrix::from_data(data, rows, cols)
    }

    // ---- reprezentacja ----
    fn __repr__(&self) -> String {
        format!("Matrix({}x{})", self.rows, self.cols)
    }

    fn __str__(&self) -> String {
        format!("{}", self)
    }

    // ---- właściwości ----
    #[getter]
    fn rows(&self) -> usize { self.rows }

    #[getter]
    fn cols(&self) -> usize { self.cols }

    // ---- operatory ----
    fn __matmul__(&self, other: &Matrix) -> PyResult<Matrix> {
        Ok(matmul(self, other)?)
    }

    fn __add__(&self, other: &Matrix) -> PyResult<Matrix> {
        Ok((self + other)?)
    }

    // ---- indeksowanie ----
    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        let (r, c) = idx;
        if r >= self.rows || c >= self.cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
        }
        Ok(self[(r, c)])
    }

    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        let (r, c) = idx;
        if r >= self.rows || c >= self.cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
        }
        self[(r, c)] = value;
        Ok(())
    }

    // ---- metody (właściwości macierzy) ----
    #[pyo3(name = "transpose")]
    fn py_transpose(&self) -> Matrix {
        self.transposed()
    }

    #[pyo3(name = "determinant")]
    fn py_determinant(&self) -> PyResult<f64> {
        Ok(self.determinant()?)
    }

    #[pyo3(name = "trace")]
    fn py_trace(&self) -> f64 {
        self.trace()
    }

    #[pyo3(name = "norm")]
    fn py_norm(&self) -> f64 {
        self.norm()
    }

    #[pyo3(name = "rank")]
    fn py_rank(&self) -> usize {
        self.rank()
    }

    #[pyo3(name = "eigenvalues")]
    fn py_eigenvalues(&self, steps: Option<u32>) -> PyResult<Vec<f64>> {
        Ok(self.eigenvalues(steps)?)
    }

    #[pyo3(name = "eigenvectors")]
    fn py_eigenv_full(&self, steps: Option<u32>) -> PyResult<(Vec<f64>, Vec<Matrix>)> {
        Ok(self.eigenv_full(steps)?)
    }

    #[pyo3(name = "row")]
    fn py_row(&self, i: usize) -> PyResult<Matrix> {
        Ok(self.row_matrix(i)?)
    }

    #[pyo3(name = "col")]
    fn py_col(&self, i: usize) -> PyResult<Matrix> {
        Ok(self.col_matrix(i)?)
    }

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
}