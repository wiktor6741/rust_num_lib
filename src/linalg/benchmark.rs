use std::time::Instant;
use crate::linalg::*;

pub fn chain_benchmark(sizes: &[(usize, usize)], iters: usize) {
    // validate chain is valid
    for i in 0..sizes.len() - 1 {
        if sizes[i].1 != sizes[i + 1].0 {
            println!("Invalid chain: matrix {} cols != matrix {} rows", i, i + 1);
            return;
        }
    }

    let mut left_to_right_times = vec![0.0; iters];
    let mut optimal_times = vec![0.0; iters];

    for iter in 0..iters {
        let matrices: Vec<Matrix> = sizes.iter()
            .map(|&(rows, cols)| random_matrix(rows, cols, 0.0, 10.0))
            .collect();

        // left to right
        let start = Instant::now();
        let mut result = matrices[0].clone();
        for i in 1..matrices.len() {
            result = matmul(&result, &matrices[i]).unwrap();
        }
        left_to_right_times[iter] = start.elapsed().as_secs_f64();

        // optimal order
        let start = Instant::now();
        let chained_result = matmul_chained(&matrices).unwrap();
        optimal_times[iter] = start.elapsed().as_secs_f64();

        if !matrices_equal(&result, &chained_result, 1e-4) { println!("different results!!");}
        println!("result ({}, {})", result.rows, result.cols);
        println!("chained_result ({}, {})", chained_result.rows, chained_result.cols);
    }

    let ltr_mean = left_to_right_times.iter().sum::<f64>() / iters as f64;
    let opt_mean = optimal_times.iter().sum::<f64>() / iters as f64;

    println!("Left to right: {:.6}s", ltr_mean);
    println!("Optimal order: {:.6}s", opt_mean);
    println!("Speedup: {:.2}x", ltr_mean / opt_mean);
}

pub fn matmul_benchmark<F>(f : F, name: &str, sizes: &[usize], iters: usize)
where F: Fn(&Matrix, &Matrix) -> Option<Matrix>
{
    for size in sizes{
        let mut results = vec![0.0; iters];

        for i in 0 .. iters {
            let m1 = random_matrix(*size, *size, 0.0, 10.0);
            let m2  = random_matrix(*size, *size, 0.0, 10.0);

            let start = Instant::now();
            let om3 = f(&m1, &m2);
            results[i] = start.elapsed().as_secs_f64();
        }

        let mean = results.iter().sum::<f64>() / results.len() as f64;
        let std_dev = (results.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / results.len() as f64).sqrt();

        println!("Name {}, size {}, mean {}, std {}", name, size, mean, std_dev);
    }
}