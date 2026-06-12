use std::ops::{Index, IndexMut};
use rand::Rng;
use std::fmt;
use crate::linalg::solve::doolitle_factorize;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows:  usize,
    pub cols: usize,
}


#[derive(Debug)]
pub enum MatrixError {
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
    NotSquare { rows: usize, cols: usize },
    NotColumnVector { cols: usize },
    SingularMatrix,
    LinearlyDependentColumns
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch { expected, got } =>
                write!(f, "Dimension mismatch: expected {:?}, got {:?}", expected, got),
            MatrixError::NotSquare { rows, cols } =>
                write!(f, "Matrix is not square: {}x{}", rows, cols),
            MatrixError::NotColumnVector { cols } =>
                write!(f, "Expected column vector, got {} columns", cols),
            MatrixError::SingularMatrix =>
                write!(f, "Matrix is singular, cannot solve"),
            MatrixError::LinearlyDependentColumns =>
                write!(f, "Columns are linearly dependent")
        }
    }
}

impl std::error::Error for MatrixError {}


impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        &mut self.data[row * self.cols + col]
    }
}

pub fn random_matrix(rows: usize, cols: usize, min : f64, max : f64) -> Matrix {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (0..rows*cols).map(|_| rng.gen_range(min..max)).collect();
    Matrix{data: v, rows, cols}
}

pub fn matrices_equal(m1: &Matrix, m2: &Matrix, epsilon: f64) -> bool {
    if m1.rows != m2.rows || m1.cols != m2.cols {
        return false;
    }
    
    for i in 0..m1.rows {
        for j in 0..m1.cols {
            if (m1[(i,j)] - m2[(i,j)]).abs() > epsilon {
                return false;
            }
        }
    }
    
    true
}

pub fn zero_matrix(rows: usize, cols: usize) -> Matrix {
    let v = vec![0.0; rows*cols];
    Matrix{rows: rows, cols: cols, data: v}
}

impl Matrix {
    pub fn from_data(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        assert_eq!(data.len(), rows * cols, 
            "data length {} doesn't match {}x{}", data.len(), rows, cols);
        Matrix { data, rows, cols }
    }


    pub fn transpose(&mut self){
        let cp = self.clone();
        self.cols = cp.rows;
        self.rows = cp.cols;
        for i in 0 .. self.rows {
            for j in 0 .. self.cols {
                self[(i, j)] = cp[(j, i)];
            }
        }
    }

    pub fn transposed(&self) -> Matrix {
        let mut result = zero_matrix(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }

    fn permutation_sign(perm: &[usize]) -> f64 {
        let n = perm.len();
        let mut inversions = 0;
        for i in 0..n {
            for j in i+1..n {
                if perm[i] > perm[j] {
                    inversions += 1;
                }
            }
        }
        if inversions % 2 == 0 { 1.0 } else { -1.0 }
    }

    pub fn determinant(a: Matrix) -> Result<f64, MatrixError> {
        match doolitle_factorize(&a) {
            Ok((_, u, perm)) => {
                let n = u.rows;
                let sign = Matrix::permutation_sign(&perm);
                Ok(sign * (0..n).map(|i| u[(i, i)]).product::<f64>())
            },
            Err(MatrixError::SingularMatrix) => Ok(0.0),
            Err(e) => Err(e),  
        }
    }
}


impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j < self.cols - 1 {
                    write!(f, "{:.4}, ", self[(i, j)])?;
                } else {
                    write!(f, "{:.4}", self[(i, j)])?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrices_equal(result: &Matrix, expected: &Matrix, epsilon: f64) {
    assert!(matrices_equal(result, expected, epsilon),
        "matrices differ:\nresult:\n{}\nexpected:\n{}", result, expected);
    }

    #[test] 
    fn transpose_test(){
        let mut m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let expected_t = Matrix::from_data(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
        m.transpose();
        assert_matrices_equal(&m, &expected_t, 1e-9);
    }

    #[test]
    fn transposed_test(){
        let m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let expected_t = Matrix::from_data(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
        let t = m.transposed();
        assert_matrices_equal(&t, &expected_t, 1e-9);
    }

    #[test]
    fn determinant_3x3_test(){
        let m = Matrix::from_data(vec![3.0, 4.0, 12.0, 8.0, 3.0, 6.0, 1.0, 2.0, 6.0], 3, 3);
        assert!((Matrix::determinant(m).unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn determinant_singular_test(){
        let a = Matrix::from_data(vec![ 1.0, 2.0, 2.0, 4.0,], 2, 2);
        let det = Matrix::determinant(a).unwrap();
        assert_eq!(det, 0.0);
    }

    #[test]
    fn test_determinant_not_square() {
        let a = Matrix::from_data(vec![1.0, 2.0, 3.0], 1, 3);
        assert!(Matrix::determinant(a).is_err());
    }
}