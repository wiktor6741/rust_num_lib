use rand::Rng;
use crate::linalg::*;

pub fn normal_sample(n: usize, mu: f64, sigma: f64) -> Matrix {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n).map(|_| {
        let u1: f64 = rng.gen_range(0.0 .. 1.0);
        let u2: f64 = rng.gen_range(0.0 .. 1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mu + sigma * z
    }).collect();
    Matrix::from_data(data, n, 1)
}


pub fn multivariete_normal_sample(n_samples: usize, mu: &Matrix, sigma: &Matrix) -> Result<Matrix, MatrixError> {
    if mu.cols != 1 {
        return Err(MatrixError::NotColumnVector { cols: mu.cols });
    }

    let n = mu.rows;
    if sigma.rows != n || sigma.cols != n {
        return Err(MatrixError::DimensionMismatch { expected: (n, n), got: (sigma.rows, sigma.cols)})
    }

    let l = cholesky(sigma)?;

 
    let z = Matrix::random(n, n_samples, 0.0, 1.0);  
    let z_data: Vec<f64> = (0..n * n_samples)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    let z = Matrix::from_data(z_data, n, n_samples);


    let mut x = matmul(&l, &z)?;

    for s in 0..n_samples {
        for i in 0..n {
            x[(i, s)] += mu[(i, 0)];
        }
    }
    Ok(x.transposed())
}