use core::f64;
use crate::linalg::matmul::*;
use crate::linalg::matrix::*;

pub fn qr_factorize(a : &Matrix) -> Result<(Matrix, Matrix), MatrixError> {
    if a.rows < a.cols {
        return Err(MatrixError::DimensionMismatch { 
            expected: (a.cols, a.cols), 
            got: (a.rows, a.cols) 
        });
    }

    let mut q = a.clone();
    let mut r = Matrix::zeros(a.cols, a.cols);

    for i in 0 .. a.cols {
        for j in 0 .. i {
            let dot: f64 = (0 .. a.rows).map(|k| q[(k, i)] * q[(k, j)]).sum();
            r[(j, i)] = dot;
            for k in (0 .. q.rows){
                q[(k, i)] -= q[(k, j)] * dot;
            }
        }

        let norm = (0 .. q.rows).map(|k| q[(k, i)].powi(2)).sum::<f64>().sqrt();
        if norm < 1e-12 {return Err(MatrixError::LinearlyDependentColumns)}
        r[(i, i)] = norm;
        for j in 0 .. q.rows {
            q[(j, i)] /= norm;
        }

    }

    Ok((q, r))
}


fn qr_algorithm(a : &Matrix, steps: u32) -> Result<Vec<f64>, MatrixError> {
    if a.rows != a.cols {return Err(MatrixError::NotSquare { rows: a.rows, cols: a.cols })}

    let mut a = a.clone();
    for i in 0 .. steps {
        let (q, r) = qr_factorize(&a)?;
        a = matmul(&r, &q)?;
    }

    Ok((0 .. a.rows).map(|i| a[(i, i)]).collect())
}

fn qr_algorithm_extended(a : &Matrix, steps: u32) -> Result<(Vec<f64>, Matrix, Matrix), MatrixError> {
    if a.rows != a.cols {return Err(MatrixError::NotSquare { rows: a.rows, cols: a.cols })}

    let mut a = a.clone();
    let mut q_hat = Matrix::identity(a.rows);
    for i in 0 .. steps {
        let (q, r) = qr_factorize(&a)?;
        q_hat = matmul(&q_hat, &q)?;
        a = matmul(&r, &q)?;
    }

    Ok(((0 .. a.rows).map(|i| a[(i, i)]).collect(), q_hat, a))
}

pub fn eigenvectors_from_schur(t: &Matrix, q: &Matrix) -> Vec<Matrix> {
    let n = t.rows;
    let mut eigenvectors = Vec::new();
    
    for i in 0..n {
        let lambda = t[(i, i)];
        let mut y = Matrix::zeros(n, 1);
        y[(i, 0)] = 1.0;
        
        for row in (0..i).rev() {
            let mut sum = t[(row, i)]; 
            for col in row+1..i {
                sum += t[(row, col)] * y[(col, 0)];
            }
            y[(row, 0)] = -sum / (t[(row, row)] - lambda);
        }
        
        let v = matmul(q, &y).unwrap();
        eigenvectors.push(v);
    }
    
    eigenvectors
}

pub fn is_symmetric(a: &Matrix, epsilon: f64) -> bool {
    if a.rows != a.cols { return false; }
    for i in 0..a.rows {
        for j in i+1..a.cols {
            if (a[(i,j)] - a[(j,i)]).abs() > epsilon {
                return false;
            }
        }
    }
    true
}

pub fn eigenvalues(a : &Matrix, steps: Option<u32>) -> Result<Vec<f64>, MatrixError> {
    let steps = steps.unwrap_or(100);
    qr_algorithm(a, steps)
}

pub fn eigenvalues_eigenvectors(a : &Matrix, steps: Option<u32>) -> Result<(Vec<f64>, Vec<Matrix>), MatrixError> {
    let symmetric = is_symmetric(a, 1e-9);
    let steps = steps.unwrap_or(100);
    let (eigenvalues, q_hat, a) = qr_algorithm_extended(a, steps)?;
    if symmetric {
        let eigenvectors : Vec<Matrix> = (0 .. q_hat.rows).map(|i| q_hat.col_matrix(i).unwrap()).collect();
        Ok((eigenvalues, eigenvectors))
    }
    else {
        Ok((eigenvalues, eigenvectors_from_schur(&a, &q_hat)))
    }
}


#[cfg(test)]
mod qr_tests {

use super::*;

    fn assert_close(a: f64, b: f64, epsilon: f64) {
        assert!((a - b).abs() < epsilon, "expected {} got {}", b, a);
    }

    fn assert_matrices_equal(result: &Matrix, expected: &Matrix, epsilon: f64) {
        assert!(matrices_equal(result, expected, epsilon),
            "matrices differ:\nresult:\n{}\nexpected:\n{}", result, expected);
    }
    // ---------- QR Factorize ----------

    #[test]
    fn test_qr_factorize_identity() {
        let a = Matrix::from_data(vec![
            1.0, 0.0,
            0.0, 1.0,
        ], 2, 2);

        let (q, r) = qr_factorize(&a).unwrap();
        assert_matrices_equal(&q, &a, 1e-9);
        assert_matrices_equal(&r, &a, 1e-9);
    }

    #[test]
    fn test_qr_factorize_reconstruction() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            3.0, 4.0,
        ], 2, 2);

        let (q, r) = qr_factorize(&a).unwrap();
        let reconstructed = matmul_naive(&q, &r).unwrap();
        assert_matrices_equal(&reconstructed, &a, 1e-9);
    }

    #[test]
    fn test_qr_factorize_orthonormal_columns() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ], 3, 2);

        let (q, _) = qr_factorize(&a).unwrap();

        // Q^T * Q powinno być I (2x2 bo Q ma 2 kolumny)
        let mut qt = q.clone();
        qt.transpose();
        let qtq = matmul_naive(&qt, &q).unwrap();

        let identity = Matrix::from_data(vec![
            1.0, 0.0,
            0.0, 1.0,
        ], 2, 2);

        assert_matrices_equal(&qtq, &identity, 1e-9);
    }

    #[test]
    fn test_qr_factorize_r_upper_triangular() {
        let a = Matrix::from_data(vec![
            2.0, 1.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ], 3, 3);

        let (_, r) = qr_factorize(&a).unwrap();

        // sprawdź zera pod przekątną
        for i in 0..3 {
            for j in 0..i {
                assert_close(r[(i, j)], 0.0, 1e-9);
            }
        }
    }

    #[test]
    fn test_qr_factorize_linearly_dependent_columns() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            2.0, 4.0,  // druga kolumna = 2 * pierwsza
        ], 2, 2);

        assert!(matches!(qr_factorize(&a), Err(MatrixError::LinearlyDependentColumns)));
    }

    #[test]
    fn test_qr_factorize_not_enough_rows() {
        let a = Matrix::from_data(vec![
            1.0, 2.0, 3.0,
        ], 1, 3);

        assert!(qr_factorize(&a).is_err());
    }

}


#[cfg(test)]
mod eigen_tests {
    use super::*;

    fn assert_matrices_equal(result: &Matrix, expected: &Matrix, epsilon: f64) {
        assert!(matrices_equal(result, expected, epsilon),
            "matrices differ:\nresult:\n{}\nexpected:\n{}", result, expected);
    }

    fn assert_close(a: f64, b: f64, epsilon: f64) {
        assert!((a - b).abs() < epsilon, "expected {} got {}", b, a);
    }

    fn assert_vec_close(a: &[f64], b: &[f64], epsilon: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_close(*x, *y, epsilon);
        }
    }

    // ---------- eigenvalues ----------

    #[test]
    fn test_eigenvalues_diagonal() {
        let a = Matrix::from_data(vec![
            2.0, 0.0,
            0.0, 5.0,
        ], 2, 2);

        let mut vals = eigenvalues(&a, None).unwrap();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());

        assert_close(vals[0], 2.0, 1e-6);
        assert_close(vals[1], 5.0, 1e-6);
    }

    #[test]
    fn test_eigenvalues_symmetric_2x2() {
        // trace=7, det=11, λ = (7±sqrt(5))/2
        let a = Matrix::from_data(vec![
            4.0, 1.0,
            1.0, 3.0,
        ], 2, 2);

        let mut vals = eigenvalues(&a, Some(100)).unwrap();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let sqrt5 = 5.0_f64.sqrt();
        assert_close(vals[0], (7.0 - sqrt5) / 2.0, 1e-6);
        assert_close(vals[1], (7.0 + sqrt5) / 2.0, 1e-6);
    }

    #[test]
    fn test_eigenvalues_sum_equals_trace() {
        let a = Matrix::from_data(vec![
            4.0, 1.0, 2.0,
            1.0, 3.0, 0.0,
            2.0, 0.0, 5.0,
        ], 3, 3);

        let vals = eigenvalues(&a, Some(200)).unwrap();
        let sum: f64 = vals.iter().sum();
        let trace = a[(0,0)] + a[(1,1)] + a[(2,2)];

        assert_close(sum, trace, 1e-6);
    }

    #[test]
    fn test_eigenvalues_not_square() {
        let a = Matrix::from_data(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ], 2, 3);

        assert!(eigenvalues(&a, None).is_err());
    }

    // ---------- eigenvectors (symmetric) ----------

    #[test]
    fn test_eigenvectors_symmetric_satisfy_av_eq_lambda_v() {
        let a = Matrix::from_data(vec![
            4.0, 1.0,
            1.0, 3.0,
        ], 2, 2);

        let (vals, vecs) = eigenvalues_eigenvectors(&a, Some(100)).unwrap();

        // sprawdź A*v_i ≈ λ_i * v_i dla każdej pary
        for (i, v) in vecs.iter().enumerate() {
            let av = matmul(&a, v).unwrap();
            for r in 0..av.rows {
                assert_close(av[(r, 0)], vals[i] * v[(r, 0)], 1e-6);
            }
        }
    }

    #[test]
    fn test_eigenvectors_symmetric_are_orthonormal() {
        let a = Matrix::from_data(vec![
            2.0, 1.0, 0.0,
            1.0, 2.0, 1.0,
            0.0, 1.0, 2.0,
        ], 3, 3);

        let (_, vecs) = eigenvalues_eigenvectors(&a, Some(200)).unwrap();

        // każdy wektor ma normę 1
        for v in &vecs {
            let norm: f64 = (0..v.rows).map(|i| v[(i,0)].powi(2)).sum::<f64>().sqrt();
            assert_close(norm, 1.0, 1e-6);
        }

        // różne wektory są ortogonalne
        for i in 0..vecs.len() {
            for j in i+1..vecs.len() {
                let dot: f64 = (0..vecs[i].rows)
                    .map(|r| vecs[i][(r,0)] * vecs[j][(r,0)])
                    .sum();
                assert_close(dot, 0.0, 1e-6);
            }
        }
    }

    #[test]
    fn test_eigenvectors_identity_matrix() {
        // dla I, każdy wektor standardowy jest wektorem własnym z λ=1
        let a = Matrix::identity(3);

        let (vals, _vecs) = eigenvalues_eigenvectors(&a, Some(50)).unwrap();

        for v in &vals {
            assert_close(*v, 1.0, 1e-6);
        }
    }

    // ---------- is_symmetric ----------

    #[test]
    fn test_is_symmetric_true() {
        let a = Matrix::from_data(vec![
            2.0, 1.0,
            1.0, 3.0,
        ], 2, 2);
        assert!(is_symmetric(&a, 1e-9));
    }

    #[test]
    fn test_is_symmetric_false() {
        let a = Matrix::from_data(vec![
            2.0, 1.0,
            5.0, 3.0,
        ], 2, 2);
        assert!(!is_symmetric(&a, 1e-9));
    }

    #[test]
    fn test_is_symmetric_not_square() {
        let a = Matrix::from_data(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ], 2, 3);
        assert!(!is_symmetric(&a, 1e-9));
    }
}