use crate::linalg::*;

pub fn mean(m : &Matrix) -> f64 {
    m.data.iter().sum::<f64>() / m.data.len() as f64
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_test(){
        let m = Matrix::from_data(vec![6.0, 17.0, 5.0, 44.0], 2, 2);
        let mean_expected = (6.0 + 17.0 + 5.0 + 44.0) / 4.0;
        assert_eq!(mean(&m), mean_expected);
    }

    #[test]
    fn mean_axis_test(){
        let m = Matrix::from_data(vec![6.0, 17.0, 5.0, 44.0, 67.0, 102.0], 3, 2);

        let expected_col_mean = Matrix::from_data(vec![(6.0 + 5.0 + 67.0) / 3.0, (17.0 + 44.0 + 102.0) / 3.0], 1, 2);
        assert_matrices_equal(&apply_along_axis(&m, mean, 0).unwrap(), &expected_col_mean, 1e-9);

        let expected_row_mean = Matrix::from_data(vec![11.5, 24.5, 84.5], 3, 1);
        assert_matrices_equal(&apply_along_axis(&m, mean, 1).unwrap(), &expected_row_mean, 1e-9);
    }
}