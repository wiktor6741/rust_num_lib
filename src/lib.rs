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
    impl From<MatrixError> for PyErr {
        fn from(e: MatrixError) -> PyErr {
            PyValueError::new_err(e.to_string())
        }
    }

    #[pymodule]
    #[pyo3(name = "rust_num_lib")]
    fn rust_num_lib(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        let linalg_mod = PyModule::new(py, "linalg")?;
        register_linalg(&linalg_mod)?;
        m.add_submodule(&linalg_mod)?;


        let stats_mod = PyModule::new(py, "stats")?;
        register_stats(&stats_mod)?;
        m.add_submodule(&stats_mod)?;
        py.import("sys")?
            .getattr("modules")?
            .set_item("rust_num_lib.stats", &stats_mod)?;



        py.import("sys")?
            .getattr("modules")?
            .set_item("rust_num_lib.linalg", &linalg_mod)?;

        Ok(())
    }

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

    #[pyfunction]
    #[pyo3(name = "matrix")]
    fn py_matrix(data: Vec<Vec<f64>>) -> PyResult<Matrix> {
        let rows = data.len();
        if rows == 0 {
            return Err(PyValueError::new_err("empty matrix"));
        }
        let cols = data[0].len();
        
        // sprawdź że wszystkie wiersze mają tę samą długość
        for row in &data {
            if row.len() != cols {
                return Err(PyValueError::new_err("rows have different lengths"));
            }
        }
        
        // spłaszcz do Vec<f64>
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        
        Ok(Matrix::from_data(flat, rows, cols))
    }

    #[pyfunction]
    #[pyo3(name = "vector")]
    fn py_vector(data : Vec<f64>) -> PyResult<Matrix> {
        Ok(linalg::vector(data))
    }

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

    #[pyfunction]
    #[pyo3(name = "cholesky")]
    fn py_cholesky(a: &Matrix) -> PyResult<Matrix> {
        Ok(linalg::cholesky(a)?)
    }



    // nowa funkcja:
    fn register_stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_mean, m)?)?;
        m.add_function(wrap_pyfunction!(py_median, m)?)?;
        m.add_function(wrap_pyfunction!(py_std_dev, m)?)?;
        m.add_function(wrap_pyfunction!(py_variance, m)?)?;
        m.add_function(wrap_pyfunction!(py_normal_sample, m)?)?;
        m.add_function(wrap_pyfunction!(py_multivariate_normal_sample, m)?)?;
        Ok(())
    }

    // wolne funkcje stats:
    #[pyfunction]
    #[pyo3(name = "mean")]
    fn py_mean(m: &Matrix) -> PyResult<f64> {
        Ok(stats::mean(m))
    }

    #[pyfunction]
    #[pyo3(name = "median")]
    fn py_median(m: &Matrix) -> PyResult<f64> {
        Ok(stats::median(m))
    }

    #[pyfunction]
    #[pyo3(name = "std_dev")]
    fn py_std_dev(m: &Matrix) -> PyResult<f64> {
        Ok(stats::std_dev(m))
    }

    #[pyfunction]
    #[pyo3(name = "variance")]
    fn py_variance(m: &Matrix) -> PyResult<f64> {
        Ok(stats::variance(m))
    }

    #[pyfunction]
    #[pyo3(name = "normal_sample")]
    fn py_normal_sample(n: usize, mu: f64, sigma: f64) -> PyResult<Matrix> {
        Ok(stats::normal_sample(n, mu, sigma))
    }

    #[pyfunction]
    #[pyo3(name = "multivariate_normal_sample")]
    fn py_multivariate_normal_sample(n_samples: usize, mu: &Matrix, sigma: &Matrix) -> PyResult<Matrix> {
        Ok(stats::multivariete_normal_sample(n_samples, mu, sigma)?)
    }
}