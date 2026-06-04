use std::time::Instant;
use rayon::prelude::*;
mod linalg;
use linalg::*;
fn main() {
    let a = linalg::random_matrix(2, 3, 0.0, 10.0);
    let b = linalg::random_matrix(3, 2, 0.0, 10.0);
    let c = linalg::random_matrix(2, 3, 0.0, 10.0);

    let r1 = matmul_tiled_parallel2(
        &matmul_tiled_parallel2(&a, &b).unwrap(), 
        &c
    ).unwrap();

    let r2 = matmul_chained(&[a, b, c]).unwrap();

    println!("{}", matrices_equal(&r1, &r2, 1e-6));



    let m1 = linalg::random_matrix(100, 500, 0.0, 10.0);
    let m2 = linalg::random_matrix(500, 100, 0.0, 10.0);
    let r1 = linalg::matmul_naive(&m1, &m2).unwrap();
    let r2 = linalg::matmul_tiled_parallel2(&m1, &m2).unwrap();
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
    linalg::matmul_benchmark(linalg::matmul_tiled_parallel2, "tiled_parallel2", &sizes, 1);
    linalg::matmul_benchmark(linalg::matmul_tiled_parallel, "tiled_parallel", &sizes, 1);
}



