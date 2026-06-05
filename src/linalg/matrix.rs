use std::ops::{Index, IndexMut};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows:  usize,
    pub cols: usize,
}


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