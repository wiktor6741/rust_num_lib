use std::time::Instant;
use crate::linalg::{matrix::*, eigenv::*, matmul::*, solve::*};


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
            .map(|&(rows, cols)| Matrix::random(rows, cols, 0.0, 10.0))
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
where F: Fn(&Matrix, &Matrix) -> Result<Matrix, MatrixError>
{
    for size in sizes{
        let mut results = vec![0.0; iters];

        for i in 0 .. iters {
            let m1 = Matrix::random(*size, *size, 0.0, 10.0);
            let m2  = Matrix::random(*size, *size, 0.0, 10.0);

            let start = Instant::now();
            let om3 = f(&m1, &m2);
            results[i] = start.elapsed().as_secs_f64();
        }

        let mean = results.iter().sum::<f64>() / results.len() as f64;
        let std_dev = (results.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / results.len() as f64).sqrt();

        println!("Name {}, size {}, mean {}, std {}", name, size, mean, std_dev);
    }
}



pub fn eigen_benchmark(sizes: &[usize], steps: u32, iters: usize) {
    for &size in sizes {
        let mut times = vec![0.0; iters];

        for i in 0..iters {
            // losowa macierz symetryczna (A + A^T jest symetryczna)
            let a = Matrix::random(size, size, 0.0, 10.0);
            let mut at = a.clone();
            at.transpose();
            let symmetric = (&a + &at).unwrap();

            let start = Instant::now();
            let _ = eigenvalues(&symmetric, Some(steps));
            times[i] = start.elapsed().as_secs_f64();
        }

        let mean = times.iter().sum::<f64>() / iters as f64;
        let std_dev = (times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / iters as f64).sqrt();

        println!("eigenvalues, size {}, steps {}, mean {:.6}, std {:.6}", size, steps, mean, std_dev);
    }
}

pub fn eigen_full_benchmark(sizes: &[usize], steps: u32, iters: usize) {
    for &size in sizes {
        let mut times = vec![0.0; iters];

        for i in 0..iters {
            let a = Matrix::random(size, size, 0.0, 10.0);
            let mut at = a.clone();
            at.transpose();
            let symmetric = (&a + &at).unwrap();

            let start = Instant::now();
            let _ = eigenvalues_eigenvectors(&symmetric, Some(steps));
            times[i] = start.elapsed().as_secs_f64();
        }

        let mean = times.iter().sum::<f64>() / iters as f64;
        let std_dev = (times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / iters as f64).sqrt();

        println!("eigenvalues+eigenvectors, size {}, steps {}, mean {:.6}, std {:.6}", size, steps, mean, std_dev);
    }
}

pub fn qr_steps_benchmark(size: usize, steps_list: &[u32], iters: usize) {
    for &steps in steps_list {
        let mut times = vec![0.0; iters];

        for i in 0..iters {
            let a = Matrix::random(size, size, 0.0, 10.0);
            let mut at = a.clone();
            at.transpose();
            let symmetric = (&a +&at).unwrap();

            let start = Instant::now();
            let _ = eigenvalues(&symmetric, Some(steps));
            times[i] = start.elapsed().as_secs_f64();
        }

        let mean = times.iter().sum::<f64>() / iters as f64;
        println!("size {}, steps {}, mean {:.6}", size, steps, mean);
    }
}


pub fn solve_benchmark(sizes: &[usize], iters: usize) {
    for &size in sizes {
        let mut gauss_times = vec![0.0; iters];
        let mut doolittle_times = vec![0.0; iters];

        for i in 0..iters {
            let a = Matrix::random(size, size, 0.0, 10.0);
            let b = Matrix::random(size, 1, 0.0, 10.0);

            let start = Instant::now();
            let _ = gauss_elimination(&a, &b);
            gauss_times[i] = start.elapsed().as_secs_f64();

            let start = Instant::now();
            let _ = doolitle_solve(&a, &b);
            doolittle_times[i] = start.elapsed().as_secs_f64();
        }

        let gauss_mean = gauss_times.iter().sum::<f64>() / iters as f64;
        let gauss_std = (gauss_times.iter().map(|x| (x - gauss_mean).powi(2)).sum::<f64>() / iters as f64).sqrt();

        let doolittle_mean = doolittle_times.iter().sum::<f64>() / iters as f64;
        let doolittle_std = (doolittle_times.iter().map(|x| (x - doolittle_mean).powi(2)).sum::<f64>() / iters as f64).sqrt();

        println!("size {}", size);
        println!("  gauss:     mean {:.6}s, std {:.6}s", gauss_mean, gauss_std);
        println!("  doolittle: mean {:.6}s, std {:.6}s", doolittle_mean, doolittle_std);
        println!("  speedup:   {:.2}x", gauss_mean / doolittle_mean);
    }
}

pub fn solve_multiple_rhs_benchmark(size: usize, rhs_counts: &[usize], iters: usize) {
    println!("Solving {}x{} system with multiple right-hand sides:", size, size);
    
    for &n_rhs in rhs_counts {
        let mut gauss_times = vec![0.0; iters];
        let mut doolittle_times = vec![0.0; iters];

        for i in 0..iters {
            let a = Matrix::random(size, size, 0.0, 10.0);
            let rhs: Vec<Matrix> = (0..n_rhs)
                .map(|_| Matrix::random(size, 1, 0.0, 10.0))
                .collect();

            // gauss — każde b osobno
            let start = Instant::now();
            for b in &rhs {
                let _ = gauss_elimination(&a, b);
            }
            gauss_times[i] = start.elapsed().as_secs_f64();

            // doolittle — faktoryzacja raz, solve wiele razy
            let start = Instant::now();
            let (l, u, perm) = doolitle_factorize(&a).unwrap();
            for b in &rhs {
                let _ = solve_factorized(&l, &u, &b, Some(&perm));
            }
            doolittle_times[i] = start.elapsed().as_secs_f64();
        }

        let gauss_mean = gauss_times.iter().sum::<f64>() / iters as f64;
        let doolittle_mean = doolittle_times.iter().sum::<f64>() / iters as f64;

        println!("  {} rhs: gauss {:.6}s, doolittle {:.6}s, speedup {:.2}x",
            n_rhs, gauss_mean, doolittle_mean, gauss_mean / doolittle_mean);
    }
}