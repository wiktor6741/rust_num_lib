use crate::linalg::*;

pub fn median(m : &Matrix) -> f64 {
    let mut data = m.data.clone();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    if n % 2 == 0 {
        (data[n/2 - 1] + data[n/2]) / 2.0
    } else {
        data[n/2]
    }
}

#[cfg(test)]
mod tests {


    use super::*;
    
    
    #[test]
    fn median_test_odd(){
        let m = Matrix::from_data(vec![4.0, 11.0, 5.0, 3.0, 7.0, 123.0, 84.0, 67.0, 1.0], 3, 3);
        let expected_median = 7.0;
        assert_eq!(median(&m), expected_median);
    }

    #[test]
    fn median_test_even(){
        let m = Matrix::from_data(vec![4.0, 11.0, 5.0, 3.0, 7.0, 123.0, 84.0, 67.0], 4, 2);
        let expected_median = 9.0;
        assert_eq!(median(&m), expected_median);
    }

    #[test]
    fn median_axis_test(){
        let m = Matrix::from_data(vec![4.0, 11.0, 5.0, 3.0, 7.0, 12.0], 2, 3);

        let expected_col_median = Matrix::from_data(vec![3.5, 9.0, 8.5], 1, 3);
        assert_matrices_equal(&apply_along_axis(&m, median, 0).unwrap(), &expected_col_median, 1e-9);

        let expected_row_median = Matrix::from_data(vec![5.0, 7.0], 2, 1);
        assert_matrices_equal(&apply_along_axis(&m, median, 1).unwrap(), &expected_row_median, 1e-9);
    }
}