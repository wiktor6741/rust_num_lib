use std::time::Instant;
use rayon::prelude::*;
mod linalg;
fn main() {
    let m1 = linalg::random_matrix(3000, 3000, 0.0, 10.0);
    let m2 = linalg::random_matrix(3000, 3000, 0.0, 10.0);
    let r1 = linalg::matmul_parallel(&m1, &m2).unwrap();
    let r2 = linalg::matmul_tiled_parallel2(&m1, &m2).unwrap();
    println!("{}", linalg::matrices_equal(&r1, &r2, 1e-6));
    //println!("{:?}", &r1);
    //println!("{:?}", &r2);
    
    let sizes: [usize; 1] = [3000];
    linalg::matmul_benchmark(linalg::matmul_tiled_parallel2, "tiled_parallel2", &sizes, 1);
    linalg::matmul_benchmark(linalg::matmul_tiled_parallel, "tiled_parallel", &sizes, 1);
}



