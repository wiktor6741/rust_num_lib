use std::ops::{Index, IndexMut};
use rand::Rng;
use std::fmt;
use crate::linalg::{eigenv, solve::doolitle_factorize, matmul::matmul};
use std::ops::{Add, Mul};

#[cfg(feature = "python")]
use pyo3::pyclass;

#[cfg_attr(feature = "python", pyclass)] 
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
    LinearlyDependentColumns,
    SliceError,
    AxisError,
    NotPositiveDefinite,
}



pub fn apply_along_axis<F>(m : &Matrix, f : F, axis: u32) -> Result<Matrix, MatrixError>
where F: Fn(&Matrix) -> f64
 {
        match axis {
        0 => {
            let data = (0 .. m.cols).map(|i| f(&m.col_matrix(i).unwrap())).collect(); 
            Ok(Matrix::from_data(data, 1, m.cols))
        },
        1 => {
            let data = (0 .. m.rows).map(|i| f(&m.row_matrix(i).unwrap())).collect(); 
            Ok(Matrix::from_data(data, m.rows, 1))
        }
        _ => Err(MatrixError::AxisError)
    }
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
                write!(f, "Columns are linearly dependent"),
            MatrixError::SliceError => 
                write!(f, "Invalid slice"),
            MatrixError::AxisError =>
                write!(f, "Axis must be 0 or 1"),
            MatrixError::NotPositiveDefinite => 
                write!(f, "Matrix is not positive definite"),
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

fn add_matrices(m1 : &Matrix, m2 : &Matrix) -> Result<Matrix, MatrixError> {
        if m1.rows != m2.rows || m1.cols != m2.cols {
            return Err(MatrixError::DimensionMismatch { expected: (m1.rows, m1.cols), got: (m2.rows, m2.cols) })}

        let mut res = Matrix::zeros(m1.rows, m1.cols);

        for i in 0 .. m1.rows {
            for j in 0 .. m1.cols{
                res[(i, j)] = m1[(i, j)] + m2[(i, j)]
            }
        }
        Ok(res)
}   

impl Add for &Matrix {
    type Output = Result<Matrix, MatrixError>;
    
    fn add(self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        add_matrices(self, rhs)  
    }
}

impl Mul for &Matrix {
    type Output = Result<Matrix, MatrixError>;
    
    fn mul(self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        matmul(self, rhs)
    }
}

impl Mul<&Matrix> for f64 {
    type Output = Matrix;
    
    fn mul(self, m: &Matrix) -> Matrix {
        m * self  
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;
    
    fn mul(self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Matrix::from_data(data, self.rows, self.cols)
    }
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
        let mut result = Matrix::zeros(self.cols, self.rows);
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

    pub fn determinant(&self) -> Result<f64, MatrixError> {
        match doolitle_factorize(&self) {
            Ok((_, u, perm)) => {
                let n = u.rows;
                let sign = Matrix::permutation_sign(&perm);
                Ok(sign * (0..n).map(|i| u[(i, i)]).product::<f64>())
            },
            Err(MatrixError::SingularMatrix) => Ok(0.0),
            Err(e) => Err(e),  
        }
    }

    pub fn slice(&self, row_range : Option<(usize, usize)>, row_step : Option<i32>,
     col_range : Option<(usize, usize)>, col_step : Option<i32>) -> Result<Matrix, MatrixError>{

        let (row_start, row_end) = row_range.unwrap_or((0, self.rows));
        let row_step = row_step.unwrap_or(1);
        let (col_start, col_end) =col_range.unwrap_or((0, self.cols));
        let col_step = col_step.unwrap_or(1);

        let row_reverse = row_step < 0;
        let col_reverse = col_step < 0;
        let row_step : usize = row_step.abs().try_into().unwrap();
        let col_step : usize = col_step.abs().try_into().unwrap();



        if row_end >  self.rows || col_end > self.cols ||
         row_start > row_end || col_start > col_end ||
          row_step == 0 || col_step == 0  {return Err(MatrixError::SliceError)}

        let row_iter = if row_reverse {(row_start .. row_end).step_by(row_step).rev().collect::<Vec<usize>>().into_iter()}
                            else {(row_start .. row_end).step_by(row_step).collect::<Vec<usize>>().into_iter()};
        
        let col_iter = if col_reverse {(col_start .. col_end).step_by(col_step).rev().collect::<Vec<usize>>().into_iter()}
                            else {(col_start .. col_end).step_by(col_step).collect::<Vec<usize>>().into_iter()};


        let rows = row_iter.len();
        let cols = col_iter.len();
        let data: Vec<f64> = row_iter.flat_map(|r| col_iter.clone().map(move |c| (r, c))).map(|(r, c)| self[(r, c)]).collect();

        Ok(Matrix::from_data(data, rows, cols))

        }

    pub fn row_matrix(&self, i: usize) -> Result<Matrix, MatrixError> {
        if i >= self.rows {return Err(MatrixError::SliceError)}

        let data : Vec<f64> = (0 .. self.cols).map(|j| self[(i, j)]).collect();

        Ok(Matrix::from_data(data, 1, self.cols))
    }    
    
    pub fn col_matrix(&self, i: usize) -> Result<Matrix, MatrixError> {
        if i >= self.cols {return Err(MatrixError::SliceError)}

        let data : Vec<f64> = (0 .. self.rows).map(|j| self[(j, i)]).collect();

        Ok(Matrix::from_data(data, self.rows, 1))
    }  

    pub fn identity(size : usize) -> Matrix {
        let mut m = Matrix::zeros(size, size);
        for i in 0 .. size {
            m[(i, i)] = 1.0;
        }
        m
    } 

    pub fn random(rows: usize, cols: usize, min : f64, max : f64) -> Matrix {
        let mut rng = rand::thread_rng();
        let v: Vec<f64> = (0..rows*cols).map(|_| rng.gen_range(min..max)).collect();
        Matrix{data: v, rows, cols}
    }    

    pub fn eigenvalues(&self, steps : Option<u32>) -> Result<Vec<f64>, MatrixError> {
        eigenv::eigenvalues(&self, steps)
    }

    pub fn eigenv_full(&self, steps : Option<u32>) -> Result<(Vec<f64>, Vec<Matrix>), MatrixError> {
        eigenv::eigenvalues_eigenvectors(&self, steps)
    }


    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        let v = vec![0.0; rows*cols];
        Matrix{rows: rows, cols: cols, data: v}
    }

    pub fn trace(&self) -> f64 {
        (0..self.rows.min(self.cols)).map(|i| self[(i,i)]).sum()
    }
    
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x*x).sum::<f64>().sqrt()
    }
    
    pub fn rank(&self) -> usize {
        match doolitle_factorize(self) {
            Ok((_, u, _)) => {
                (0..u.rows.min(u.cols))
                    .filter(|&i| u[(i, i)].abs() > 1e-10)
                    .count()
            },
            Err(_) => 0,
        }
    }
    
    pub fn is_symmetric(&self, epsilon: f64) -> bool {
        if self.rows != self.cols { return false; }
            for i in 0..self.rows {
                for j in i+1..self.cols {
                    if (self[(i,j)] - self[(j,i)]).abs() > epsilon {
                        return false;
                    }
                }
            }
            true
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

pub fn assert_matrices_equal(result: &Matrix, expected: &Matrix, epsilon: f64) {
    assert!(matrices_equal(result, expected, epsilon),
        "matrices differ:\nresult:\n{}\nexpected:\n{}", result, expected);
    }

pub type Vector = Matrix;

pub fn vector(data: Vec<f64>) -> Vector {
        let n = data.len();
        Matrix::from_data(data, n, 1)
}

impl Index<usize> for Matrix {
    type Output = f64;
    
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}


#[cfg(test)]
mod tests {
    use super::*;


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
        assert!((m.determinant().unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn determinant_singular_test(){
        let a = Matrix::from_data(vec![ 1.0, 2.0, 2.0, 4.0,], 2, 2);
        let det = a.determinant().unwrap();
        assert_eq!(det, 0.0);
    }

    #[test]
    fn test_determinant_not_square() {
        let a = Matrix::from_data(vec![1.0, 2.0, 3.0], 1, 3);
        assert!(a.determinant().is_err());
    }

    #[test]
    fn test_slice_basic() {
        let m = Matrix::from_data((0..20).map(|x| x as f64).collect(), 4, 5);
        // m:
        //  0  1  2  3  4
        //  5  6  7  8  9
        // 10 11 12 13 14
        // 15 16 17 18 19
        
        // wiersze 1-3, kolumny 1-4, krok 1
        let s = m.slice(Some((1, 3)), None, Some((1, 4)), None).unwrap();
        // powinno być:
        //  6  7  8
        // 11 12 13
        assert_eq!(s.rows, 2);
        assert_eq!(s.cols, 3);
        assert_eq!(s[(0,0)], 6.0);
        assert_eq!(s[(1,2)], 13.0);
    }

    #[test]
    fn test_slice_with_step() {
        let m = Matrix::from_data((0..20).map(|x| x as f64).collect(), 4, 5);
        
        // wiersze 0,2 (krok 2), wszystkie kolumny
        let s = m.slice(Some((0, 4)), Some(2), None, None).unwrap();
        assert_eq!(s.rows, 2);
        assert_eq!(s[(0,0)], 0.0);
        assert_eq!(s[(1,0)], 10.0);
    }

    #[test]
    fn test_slice_reverse() {
        let m = Matrix::from_data((0..20).map(|x| x as f64).collect(), 4, 5);
        
        // wiersze odwrócone
        let s = m.slice(Some((0, 4)), Some(-1), None, None).unwrap();
        assert_eq!(s[(0,0)], 15.0);  // ostatni wiersz pierwszy
        assert_eq!(s[(3,0)], 0.0);   // pierwszy wiersz ostatni
    }

    #[test]
    fn test_slice_invalid_range() {
        let m = Matrix::from_data((0..20).map(|x| x as f64).collect(), 4, 5);
        
        assert!(m.slice(Some((3, 1)), None, None, None).is_err());  // start > end
        assert!(m.slice(Some((0, 10)), None, None, None).is_err()); // poza zakresem
        assert!(m.slice(None, Some(0), None, None).is_err());       // step = 0
    }
}