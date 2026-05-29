use crate::linalg::{matmul_better, matmul_naive, matmul_parallel_simple, matmul_tiled_parallel, matmul_tiled_parallel2, random_matrix};
use std::time::Instant;
use rayon::prelude::*;
mod linalg;
fn main() {
    let m1 = random_matrix(2000, 2000, 0.0, 10.0);
    let m2 = random_matrix(2000, 2000, 0.0, 10.0);

    let cols = m2.cols;
    let mut data = vec![0.0; m1.rows * cols];

    let start = Instant::now();
    data.chunks_mut(cols)
        .enumerate()
        .for_each(|(i, row)| {
            for k in 0..m1.cols {
                for j in 0..cols {
                    row[j] += m1.data[i * m1.cols + k] * m2.data[k * cols + j];
                }
            }
        });
    println!("chunks_mut sequential: {:.3}s", start.elapsed().as_secs_f64());


    for threads in [1, 2, 4, 8] {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            let m1 = random_matrix(2000, 2000, 0.0, 10.0);
            let m2 = random_matrix(2000, 2000, 0.0, 10.0);
            
            let cols = m2.cols;
            let mut data = vec![0.0; m1.rows * cols];
            
            let start = Instant::now();
            data.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for k in 0..m1.cols {
                        for j in 0..cols {
                            row[j] += m1.data[i * m1.cols + k] * m2.data[k * cols + j];
                        }
                    }
                });
            println!("{} threads: {:.3}s", threads, start.elapsed().as_secs_f64());
        });
    }
    let data: Vec<u64> = (0..10_000_000u64).map(|x| x * 3 + 7).collect();

    let start_seq = Instant::now();
    let sum_seq: u64 = data.iter().sum();
    let elapesed_seq = start_seq.elapsed().as_secs_f64();

    let start_par = Instant::now();
    let sum_par: u64 = data.into_par_iter().sum();
    let elapsed_par = start_par.elapsed().as_secs_f64();

    println!("sequential: {}, time: {}", sum_seq, elapesed_seq);
    println!("parallel: {}, time: {}", sum_par, elapsed_par);


    println!("Rayon threads: {}", rayon::current_num_threads());
    let sizes: [usize; 1] = [2000];

    let m1 = linalg::random_matrix(100, 100, 0.0, 10.0);
    let m2 = linalg::random_matrix(100, 100, 0.0, 10.0);
    let eps = 1e-9;
    let res1 = linalg::matmul_naive(&m1, &m2).unwrap();
    let res2 = linalg::matmul_tiled_parallel(&m1, &m2).unwrap();

    println!("{}", linalg::matrices_equal(&res1, &res2, eps));
    linalg::matmul_benchmark(matmul_better, "better", &sizes, 1);
    linalg::matmul_benchmark(matmul_tiled_parallel2, "tiled_parallel", &sizes, 1);
}



