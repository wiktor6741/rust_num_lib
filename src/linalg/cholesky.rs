use crate::linalg::*;

pub fn cholesky(a: &Matrix) -> Result<Matrix, MatrixError> {
    if a.rows != a.cols {
        return Err(MatrixError::NotSquare { rows: a.rows, cols: a.cols });
    }

    let n = a.rows;
    let mut l = Matrix::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[(i, k)] * l[(j, k)]).sum();

            if i == j {
                let val = a[(i, i)] - sum;
                if val < 0.0 {
                    return Err(MatrixError::NotPositiveDefinite);
                }
                l[(i, j)] = val.sqrt();
            } else {
                if l[(j, j)].abs() < 1e-12 {
                    return Err(MatrixError::SingularMatrix);
                }
                l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
            }
        }
    }

    Ok(l)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_2x2() {
        let a = Matrix::from_data(vec![
            4.0, 2.0,
            2.0, 2.0,
        ], 2, 2);

        let l = cholesky(&a).unwrap();
        
        let mut lt = l.clone();
        lt.transpose();
        let reconstructed = matmul_naive(&l, &lt).unwrap();
        assert_matrices_equal(&reconstructed, &a, 1e-9);
        
        assert!((l[(0, 1)]).abs() < 1e-9);
    }

    #[test]
    fn test_cholesky_3x3() {
        let a = Matrix::from_data(vec![
            4.0, 2.0, 2.0,
            2.0, 3.0, 1.0,
            2.0, 1.0, 3.0,
        ], 3, 3);

        let l = cholesky(&a).unwrap();

        // L*L^T = A
        let mut lt = l.clone();
        lt.transpose();
        let reconstructed = matmul_naive(&l, &lt).unwrap();
        assert_matrices_equal(&reconstructed, &a, 1e-9);

        // L dolnotrójkątna — zera powyżej przekątnej
        for i in 0..3 {
            for j in i+1..3 {
                assert!((l[(i, j)]).abs() < 1e-9,
                    "L[{},{}] = {} powinno być 0", i, j, l[(i, j)]);
            }
        }
    }

    #[test]
    fn test_cholesky_identity() {
        // Cholesky(I) = I
        let a = Matrix::identity(3);
        let l = cholesky(&a).unwrap();
        assert_matrices_equal(&l, &a, 1e-9);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            2.0, 1.0,  
        ], 2, 2);

        assert!(matches!(cholesky(&a), Err(MatrixError::NotPositiveDefinite)));
    }

    #[test]
    fn test_cholesky_not_square() {
        let a = Matrix::from_data(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ], 2, 3);

        assert!(matches!(cholesky(&a), Err(MatrixError::NotSquare { .. })));
    }

    #[test]
    fn test_cholesky_diagonal() {
        let a = Matrix::from_data(vec![
            4.0, 0.0, 0.0,
            0.0, 9.0, 0.0,
            0.0, 0.0, 16.0,
        ], 3, 3);

        let l = cholesky(&a).unwrap();

        assert!((l[(0,0)] - 2.0).abs() < 1e-9);
        assert!((l[(1,1)] - 3.0).abs() < 1e-9);
        assert!((l[(2,2)] - 4.0).abs() < 1e-9);
    }
}