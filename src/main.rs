use crate::linalg::{matmul_better, matmul_naive, matmul_tiled_parallel};

mod linalg;
fn main() {
    let sizes: [usize; 1] = [2000];

    let m1 = linalg::random_matrix(100, 100, 0.0, 10.0);
    let m2 = linalg::random_matrix(100, 100, 0.0, 10.0);
    let eps = 1e-9;
    let res1 = linalg::matmul_naive(&m1, &m2).unwrap();
    let res2 = linalg::matmul_tiled_parallel(&m1, &m2).unwrap();

    println!("{}", linalg::matrices_equal(&res1, &res2, eps));

    linalg::matmul_benchmark(matmul_naive, "naive", &sizes, 1);
    linalg::matmul_benchmark(matmul_tiled_parallel, "tiled_parallel", &sizes, 1);
}



