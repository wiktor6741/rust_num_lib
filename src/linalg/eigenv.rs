use core::f64;
use crate::linalg::matmul::*;
use crate::linalg::matrix::*;

fn qr_factorize(a : &Matrix) -> Result<(Matrix, Matrix), MatrixError> {
    if a.rows < a.cols {
        return Err(MatrixError::DimensionMismatch { 
            expected: (a.cols, a.cols), 
            got: (a.rows, a.cols) 
        });
    }

    let mut q = a.clone();
    let mut r = zero_matrix(a.cols, a.cols);

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
    println!("{:?}", a);
    Ok((0 .. a.rows).map(|i| a[(i, i)]).collect())
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

    // ---------- QR Algorithm (eigenvalues) ----------

    #[test]
    fn test_qr_eigenvalues_diagonal() {
        // wartości własne to po prostu elementy przekątnej
        let a = Matrix::from_data(vec![
            2.0, 0.0,
            0.0, 5.0,
        ], 2, 2);

        let mut eigenvalues = qr_algorithm(&a, 50).unwrap();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_close(eigenvalues[0], 2.0, 1e-6);
        assert_close(eigenvalues[1], 5.0, 1e-6);
    }

    #[test]
    fn test_qr_eigenvalues_symmetric_2x2() {
        // trace = 7, det = 4*3 - 1*2 = 10
        // λ = (7 ± sqrt(49-40))/2 = (7±3)/2 = {5, 2}
        let a = Matrix::from_data(vec![
            3.0, 2.0_f64.sqrt(),
            2.0_f64.sqrt(), 4.0,
        ], 2, 2);

        let mut eigenvalues = qr_algorithm(&a, 200).unwrap();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        //println!("aaaa");
        assert_close(eigenvalues[0], 2.0, 1e-6);
        //println!("bbbbbb");
        assert_close(eigenvalues[1], 5.0, 1e-6);
    }

    #[test]
    fn test_qr_eigenvalues_symmetric_3x3() {
        // macierz symetryczna, wartości własne policzone niezależnie
        let a = Matrix::from_data(vec![
            2.0, 1.0, 0.0,
            1.0, 2.0, 1.0,
            0.0, 1.0, 2.0,
        ], 3, 3);
        // wartości własne: 2, 2+sqrt(2), 2-sqrt(2)

        let mut eigenvalues = qr_algorithm(&a, 200).unwrap();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sqrt2 = 2.0_f64.sqrt();
        assert_close(eigenvalues[0], 2.0 - sqrt2, 1e-6);
        assert_close(eigenvalues[1], 2.0, 1e-6);
        assert_close(eigenvalues[2], 2.0 + sqrt2, 1e-6);
    }

    #[test]
    fn test_qr_eigenvalues_sum_equals_trace() {
        // suma wartości własnych = trace (zachowane przez QR algorithm)
        let a = Matrix::from_data(vec![
            4.0, 1.0, 2.0,
            1.0, 3.0, 0.0,
            2.0, 0.0, 5.0,
        ], 3, 3);

        let eigenvalues = qr_algorithm(&a, 200).unwrap();
        let sum: f64 = eigenvalues.iter().sum();
        let trace = a[(0,0)] + a[(1,1)] + a[(2,2)];

        assert_close(sum, trace, 1e-6);
    }

    #[test]
    fn test_qr_eigenvalues_not_square() {
        let a = Matrix::from_data(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ], 2, 3);

        assert!(qr_algorithm(&a, 50).is_err());
    }
}