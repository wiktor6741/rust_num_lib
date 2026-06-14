// src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

pub mod linalg;

use crate::linalg::{Matrix, MatrixError};

mod python;

// konwersja błędów: MatrixError -> Python ValueError
impl From<MatrixError> for PyErr {
    fn from(e: MatrixError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

#[pymodule]
#[pyo3(name = "linalg")] 
fn linalg_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Matrix>()?;

    m.add_function(wrap_pyfunction!(py_matmul_chained, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_multiple, m)?)?;
    m.add_function(wrap_pyfunction!(py_qr_factorize, m)?)?;
    m.add_function(wrap_pyfunction!(py_lu_factorize, m)?)?;

    Ok(())
}

// ---- wolne funkcje ----

#[pyfunction]
#[pyo3(name = "matmul_chained")]
fn py_matmul_chained(matrices: Vec<Matrix>) -> PyResult<Matrix> {
    Ok(linalg::matmul_chained(&matrices)?)
}

#[pyfunction]
#[pyo3(name = "solve")]
fn py_solve(a: &Matrix, b: &Matrix) -> PyResult<Matrix> {
    Ok(linalg::gauss_elimination(a, b)?)
}

#[pyfunction]
#[pyo3(name = "solve_multiple")]
fn py_solve_multiple(a: &Matrix, bs: Vec<Matrix>) -> PyResult<Vec<Matrix>> {
    Ok(linalg::solve_multiple(a, &bs)?)
}

#[pyfunction]
#[pyo3(name = "qr_factorize")]
fn py_qr_factorize(a: &Matrix) -> PyResult<(Matrix, Matrix)> {
    Ok(linalg::qr_factorize(a)?)
}

#[pyfunction]
#[pyo3(name = "lu_factorize")]
fn py_lu_factorize(a: &Matrix) -> PyResult<(Matrix, Matrix, Vec<usize>)> {
    Ok(linalg::doolitle_factorize(a)?)
}