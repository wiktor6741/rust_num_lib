use crate::linalg::matrix::*;

fn swap_rows(m: &mut Matrix, row1: usize, row2: usize) {
    for j in 0..m.cols {
        m.data.swap(row1 * m.cols + j, row2 * m.cols + j);
    }
}

pub fn gauss_elimination(m : &Matrix, b : &Matrix) -> Result<Matrix, MatrixError> {
    if b.cols != 1 {
        return Err(MatrixError::NotColumnVector { cols: b.cols });
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

        if m_max.abs() < 1e-12 {
            return Err(MatrixError::SingularMatrix);
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

    if m[(n - 1 , n - 1)].abs() < 1e-12 {return Err(MatrixError::SingularMatrix);}

    for i in (0 .. n).rev() {
        b[(i, 0)] /= m[(i, i)];
        for j in 0 .. i {
            b[(j, 0)] -= b[(i, 0)] * m[(j, i)];
        }
    }

    Ok(b)
}



pub fn doolitle_factorize(a: &Matrix) -> Result<(Matrix, Matrix), MatrixError>{
    if a.rows != a.cols{
        return Err(MatrixError::NotSquare { rows: (a.rows), cols: (a.cols) })
    }
    let mut n = a.rows;

    let mut l = zero_matrix(n, n);
    let mut u = zero_matrix(n, n);

    for i in 0 .. n {
        for j in i .. n {
            let s: f64 = (0..i).map(|p| l[(i, p)] * u[(p, j)]).sum();
            u[(i, j)] = a[(i, j)] - s;
            if i == j {
                l[(i, j)] = 1.0;
                if (u[(i, j)] - 0.0).abs() < 1e-12 {return Err(MatrixError::SingularMatrix);} 
            }
            else {
                let s: f64 = (0 .. i).map(|p| (l[(j, p)] * u[(p, i)])).sum();
                l[(j, i)] = (a[(j, i)] - s) / u[(i, i)];
            }
        }
    }

    Ok((l, u))
}

pub fn solve_factorized(l : &Matrix, u : &Matrix, b : &Matrix) -> Result<Matrix, MatrixError>{
    if b.cols != 1 {
        return Err(MatrixError::NotColumnVector { cols: b.cols });
    }

    let n = b.rows;

    if l.rows != n || l.cols != n {
        return Err(MatrixError::DimensionMismatch { expected: (n, n), got: (l.rows, l.cols) });
    }

    if u.rows != n || u.cols != n {
        return Err(MatrixError::DimensionMismatch { expected: (n, n), got: (u.rows, u.cols) });
    } 

    let l = l.clone();
    let u = u.clone();
    let mut b = b.clone();

    for i in 0 .. n {
        b[(i, 0)] /= l[(i, i)];
        for j in i + 1 .. n {
            b[(j, 0)] -= b[(i, 0)] * l[(j, i)];
        }
    }

    for i in (0 .. n).rev() {
        b[(i, 0)] /= u[(i, i)];
        for j in 0 .. i {
            b[(j, 0)] -= b[(i, 0)] * u[(j, i)];
        }
    }

    Ok(b)
}

pub fn doolitle_solve(m : &Matrix, b : &Matrix) -> Result<Matrix, MatrixError> {
    let (l, u) = doolitle_factorize(m)?;
    let x = solve_factorized(&l, &u, &b)?;
    Ok(x)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrices_equal(result: &Matrix, expected: &Matrix, epsilon: f64) {
    assert!(matrices_equal(result, expected, epsilon),
        "matrices differ:\nresult:\n{}\nexpected:\n{}", result, expected);
    }

    #[test]
    fn test_gauss_simple_2x2() {
        // 2x + y = 5
        // x + 3y = 10
        // rozwiązanie: x = 1, y = 3
        let m = Matrix::from_data(vec![2.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Matrix::from_data(vec![5.0, 10.0], 2, 1);
        let result = gauss_elimination(&m, &b).unwrap();
        let expected = Matrix::from_data(vec![1.0, 3.0], 2, 1);
        assert_matrices_equal(&result, &expected, 1e-9);
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
        let expected = Matrix::from_data(vec![5.0, 3.0, -2.0], 3, 1);
        assert_matrices_equal(&result, &expected, 1e-9);
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
        assert_matrices_equal(&result, &b, 1e-9);
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

    #[test]
    fn test_gauss_singular_matrix() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            2.0, 4.0,  
        ], 2, 2);
        let b = Matrix::from_data(vec![1.0, 3.0], 2, 1);
        assert!(gauss_elimination(&a, &b).is_err());
    }

    #[test]
    fn test_doolitle_3x3(){
        let a = Matrix::from_data(vec![3.0, 4.0, 1.0, 18.0, 29.0, 8.0, 21.0, 43.0, 14.0],
             3, 3);
        let (l, u) = doolitle_factorize(&a).unwrap();
        let expected_l = Matrix::from_data(vec![1.0, 0.0, 0.0, 6.0, 1.0, 0.0, 7.0, 3.0, 1.0],
             3, 3);
        assert_matrices_equal(&l, &expected_l, 1e-9);

        let expected_u = Matrix::from_data(vec![3.0, 4.0, 1.0, 0.0, 5.0, 2.0, 0.0, 0.0, 1.0],
        3, 3);
        assert_matrices_equal(&u, &expected_u, 1e-9);
    }

    #[test]
    fn test_lu_singular_matrix() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            2.0, 4.0,  
        ], 2, 2);
        assert!(doolitle_factorize(&a).is_err());
    }
    #[test]
    fn test_doolitle_solve_2x2() {
        // 2x + y = 5
        // x + 3y = 10
        // rozwiązanie: x = 1, y = 3
        let m = Matrix::from_data(vec![2.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Matrix::from_data(vec![5.0, 10.0], 2, 1);
        let result = doolitle_solve(&m, &b).unwrap();
        let expected = Matrix::from_data(vec![1.0, 3.0], 2, 1);
        assert_matrices_equal(&result, &expected, 1e-9);
    }

    #[test]
    fn test_doolitle_solve_3x3() {
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
        let result = doolitle_solve(&m, &b).unwrap();
        let expected = Matrix::from_data(vec![5.0, 3.0, -2.0], 3, 1);
        assert_matrices_equal(&result, &expected, 1e-9);
    }

    #[test]
    fn test_doolitle_solve_identity() {
        // macierz jednostkowa — rozwiązanie to samo b
        let m = Matrix::from_data(vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ], 3, 3);
        let b = Matrix::from_data(vec![3.0, 7.0, 2.0], 3, 1);
        let result = doolitle_solve(&m, &b).unwrap();
        assert_matrices_equal(&result, &b, 1e-9);
    }

    #[test]
    fn test_doolitle_solve_wrong_dimensions() {
        let m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data(vec![1.0, 2.0, 3.0], 3, 1);
        assert!(doolitle_solve(&m, &b).is_err());
    }

    #[test]
    fn test_doolitle_b_not_column_vector() {
        let m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(doolitle_solve(&m, &b).is_err());
    }

    #[test]
    fn test_doolittle_singular_matrix() {
        let a = Matrix::from_data(vec![
            1.0, 2.0,
            2.0, 4.0,  
        ], 2, 2);
        let b = Matrix::from_data(vec![1.0, 3.0], 2, 1);
        assert!(doolitle_solve(&a, &b).is_err());
    }
}



