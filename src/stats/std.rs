use crate::linalg::*;
use crate::stats::*;

pub fn variance(m : &Matrix) -> f64 {
    let n = m.data.len();
    let mu = mean(m);
    let sum : f64 = (0 .. n).map(|i| (m.data[i] - mu).powi(2)).sum();
    (sum / n as f64)
}

pub fn std_dev(m :&Matrix) -> f64 {
    variance(m).sqrt()
}