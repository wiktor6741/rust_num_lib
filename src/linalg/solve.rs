use crate::linalg::matrix::*;

fn swap_rows(m: &mut Matrix, row1: usize, row2: usize) {
    for j in 0..m.cols {
        m.data.swap(row1 * m.cols + j, row2 * m.cols + j);
    }
}

pub fn gauss_elimination(m : &Matrix, b : &Matrix) -> Result<Matrix, MatrixError> {
    if b.cols != 1 {
        return Err(MatrixError::NotColumnVector { cols: (b.cols) });
    }

    let n = b.rows;

    if m.rows != n || m.cols != n {
        return Err(MatrixError::DimensionMismatch { expected: (n, n), got: (m.rows, m.cols) });
    }

    let mut m = m.clone();
    let mut b = b.clone();

    for i in 0 .. n - 1 {
        let mut  m_max = m[(i, i)];
        let mut max_j = i;
        for j in i .. n {
            if m[(j, i)] > m_max {
                m_max = m[(j, i)];
                max_j = j;
            }
        }

        if max_j != i {
            swap_rows(&mut m, i, max_j);
            b.data.swap(i, max_j);
        }
        
        for j in i + 1 .. n {
            let factor = (m[(j, i)] / m[(i, i)]);
            for k in i .. n {
                m[(j, k)] -= m[(i, k)] * factor;
            }
            b[(j, 0)] -= b[(i, 0)] * factor;
        }  
        println!("m: {}, b: {}", m , b);
    }

    for i in (0 .. n).rev() {
        b[(i, 0)] /= m[(i, i)];
        for j in 0 .. i {
            b[(j, 0)] -= b[(i, 0)] * m[(j, i)];
        }
    }

    Ok(b)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrix_close(result: &Matrix, expected: &[f64], epsilon: f64) {
        for (i, (&r, &e)) in result.data.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < epsilon, 
                "element {} differs: got {}, expected {}", i, r, e);
        }
    }

    #[test]
    fn test_gauss_simple_2x2() {
        // 2x + y = 5
        // x + 3y = 10
        // rozwiązanie: x = 1, y = 3
        let m = Matrix::from_data(vec![2.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Matrix::from_data(vec![5.0, 10.0], 2, 1);
        let result = gauss_elimination(&m, &b).unwrap();
        assert_matrix_close(&result, &[1.0, 3.0], 1e-9);
    }

    #[test]
    fn test_gauss_3x3() {
        // x + y + z = 6
        // 2y + 5z = -4
        // 2x + 5y - z = 27
        // rozwiązanie: x = 5, y = 3, z = -2
        let m = Matrix::from_data(vec![
            1.0, 1.0,  1.0,
            0.0, 2.0,  5.0,
            2.0, 5.0, -1.0,
        ], 3, 3);
        let b = Matrix::from_data(vec![6.0, -4.0, 27.0], 3, 1);
        let result = gauss_elimination(&m, &b).unwrap();
        assert_matrix_close(&result, &[5.0, 3.0, -2.0], 1e-9);
    }

    #[test]
    fn test_gauss_identity() {
        // macierz jednostkowa — rozwiązanie to samo b
        let m = Matrix::from_data(vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ], 3, 3);
        let b = Matrix::from_data(vec![3.0, 7.0, 2.0], 3, 1);
        let result = gauss_elimination(&m, &b).unwrap();
        assert_matrix_close(&result, &[3.0, 7.0, 2.0], 1e-9);
    }

    #[test]
    fn test_gauss_wrong_dimensions() {
        let m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data(vec![1.0, 2.0, 3.0], 3, 1);
        assert!(gauss_elimination(&m, &b).is_err());
    }

    #[test]
    fn test_gauss_b_not_column_vector() {
        let m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(gauss_elimination(&m, &b).is_err());
    }
}