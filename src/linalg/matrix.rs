use std::ops::{Index, IndexMut};
use rand::Rng;
use std::fmt;

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