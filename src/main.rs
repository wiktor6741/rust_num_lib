use std::time::Instant;
use rayon::prelude::*;
mod linalg;
use linalg::*;
mod stats;
fn main() {

    linalg::solve_benchmark(&[10, 50, 100, 200, 500], 10);
    linalg::solve_multiple_rhs_benchmark(100, &[1, 5, 10, 50, 100], 5);

    linalg::eigen_benchmark(&[10, 50, 100, 200], 100, 5);
    linalg::eigen_full_benchmark(&[10, 50, 100, 200], 100, 5);
    linalg::qr_steps_benchmark(100, &[10, 50, 100, 200, 500], 5);

    let a = linalg::Matrix::random(2, 3, 0.0, 10.0);
    let b = linalg::Matrix::random(3, 2, 0.0, 10.0);
    let c = linalg::Matrix::random(2, 3, 0.0, 10.0);

    let r1 = matmul(
        &matmul(&a, &b).unwrap(), 
        &c
    ).unwrap();

    let r2 = matmul_chained(&[a, b, c]).unwrap();

    println!("{}", matrices_equal(&r1, &r2, 1e-6));



    let m1 = linalg::Matrix::random(100, 500, 0.0, 10.0);
    let m2 = linalg::Matrix::random(500, 100, 0.0, 10.0);
    let r1 = linalg::matmul_naive(&m1, &m2).unwrap();
    let r2 = linalg::matmul(&m1, &m2).unwrap();
    println!("{}", linalg::matrices_equal(&r1, &r2, 1e-6));
    //println!("{:?}", &r1);
    //println!("{:?}", &r2);

    let sizes = &[
        (1000, 1),    // 1000x1
        (1, 1000),    // 1x1000
        (1000, 1),    // 1000x1
        (1, 1000),    // 1x1000
        (1000, 1),    // 1000x1
    ];

    linalg::chain_benchmark(sizes, 5);

    
    let sizes: [usize; 1] = [3000];
    linalg::matmul_benchmark(linalg::matmul, "tiled_parallel2", &sizes, 1);
    linalg::matmul_benchmark(linalg::matmul_tiled_parallel, "tiled_parallel", &sizes, 1);
}



