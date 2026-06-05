use std::time::Instant;
use std::cmp::min;
use rayon::prelude::*;
use crate::linalg::matrix::*;


pub fn matmul_naive(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
       return None
    }

    let v: Vec<f64> = vec![0.0; m1.rows*m2.cols];

    let mut m = Matrix{data: v, rows: m1.rows, cols: m2.cols};

    for i in 0 .. m1.rows {
        for j in 0 .. m2.cols{
            for k in 0.. m1.cols{
                m[(i, j)] += m1[(i, k)] * m2[(k, j)]
            }
        }
    }   

    Some(m)
}



pub fn matmul_better(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
       return None
    }

    let v: Vec<f64> = vec![0.0; m1.rows*m2.cols];

    let mut m = Matrix{data: v, rows: m1.rows, cols: m2.cols};

    for i in 0 .. m1.rows {
        for k in 0 .. m2.rows {
            for j in 0 .. m2.cols {
                m[(i, j)] +=  m1[(i, k)] * m2[(k, j)];
            }
        }
    }   

    Some(m)
} 

pub fn matmul_tiled_parallel(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
        return None;
    }
    
    let tile = 256;
    let rows = m1.rows;
    let cols = m2.cols;
    let inner = m1.cols;
    
    let mut data = vec![0.0; rows * cols];
    
    data.par_chunks_mut(tile * cols)
        .enumerate()
        .for_each(|(i_t, chunk)| {
            let i_start = i_t * tile;
            let i_end = min(rows, i_start + tile);
            
            for k_t in 0..(inner + tile - 1) / tile {
                let k_start = k_t * tile;
                let k_end = min(inner, k_start + tile);
                
                for j_t in 0..(cols + tile - 1) / tile {
                    let j_start = j_t * tile;
                    let j_end = min(cols, j_start + tile);
                    
                    let mut a_tile = vec![0.0; tile * tile];
                    let mut b_tile = vec![0.0; tile * tile];
                    
                    for i in i_start..i_end {
                        for k in k_start..k_end {
                            a_tile[(i-i_start)*tile+(k-k_start)] = m1[(i,k)];
                        }
                    }
                    for k in k_start..k_end {
                        for j in j_start..j_end {
                            b_tile[(k-k_start)*tile+(j-j_start)] = m2[(k,j)];
                        }
                    }
                    
                    for i in 0..i_end-i_start {
                        for k in 0..k_end-k_start {
                            for j in 0..j_end-j_start {
                                chunk[i*cols+(j+j_start)] += 
                                    a_tile[i*tile+k] * b_tile[k*tile+j];
                            }
                        }
                    }
                }
            }
        });
    
    Some(Matrix { data, rows, cols })
}


pub fn matmul(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
        return None;
    }
    let tile = 256;
    let rows = m1.rows;
    let cols = m2.cols;
    let inner = m1.cols;

    let num_tile_rows = (rows + tile - 1) / tile;
    let num_threads = if num_tile_rows >= 8 {8} else {num_tile_rows};
    let high_tile_rows_per_thread = (num_tile_rows + num_threads - 1) / num_threads;
    let low_tile_rows_per_thread = high_tile_rows_per_thread - 1;
    let num_high = if num_tile_rows % num_threads == 0 {num_threads} else {num_tile_rows % num_threads};

    let mut data = vec![0.0; rows * cols];
    let data_ptr = data.as_mut_ptr() as usize;
    let m1_ptr = m1.data.as_ptr() as usize;
    let m2_ptr = m2.data.as_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..num_threads {
            let high = t + 1 <= num_high;
            let start_tile_row = if high {high_tile_rows_per_thread * t} else 
            {high_tile_rows_per_thread * num_high + low_tile_rows_per_thread * (t - num_high)};

            let local_tile_rows = if high {high_tile_rows_per_thread} else {low_tile_rows_per_thread};

            let end_tile_row = min(start_tile_row + local_tile_rows - 1, num_tile_rows - 1);
            let start_row = start_tile_row * tile;
            let end_row = min((end_tile_row + 1) * tile - 1, rows -1);
            s.spawn(move || {
               let out = unsafe { std::slice::from_raw_parts_mut(
                    (data_ptr as *mut f64).add(start_row *cols), 
                    (end_row - start_row +1) * cols
                )};

                let a = unsafe { std::slice::from_raw_parts(
                    (m1_ptr as *const f64).add(start_row * inner),
                     (end_row - start_row + 1) * inner
                )};

                let b = unsafe { std::slice::from_raw_parts(
                    m2_ptr as *const f64, inner * cols
                )};
            
                for i_t in 0 .. end_tile_row - start_tile_row + 1 {
                    let i_start = i_t * tile;
                    let i_end = min(i_start + tile, end_row - start_row + 1);
                    //println!("{}", i_end);
                    for k_t in 0 .. (inner + tile - 1) / tile {
                        let k_start = k_t * tile;
                        let k_end = min(k_start + tile, inner);

                        let mut a_tile = vec![0.0; tile * tile];

                        for i in i_start..i_end {
                            for k in k_start..k_end {
                                a_tile[(i-i_start)*tile+(k-k_start)] = a[i * inner + k];
                            }
                        }
                        
                        for j_t in 0 .. (cols + tile - 1) / tile{
                            let j_start = j_t * tile;
                            let j_end = min(j_start + tile, cols);

                            let mut b_tile = vec![0.0; tile * tile];
                            
                            for k in k_start..k_end {
                                for j in j_start..j_end {
                                    b_tile[(k-k_start)*tile+(j-j_start)] = b[k * cols + j];
                                }
                            }
                            
                            for i in i_start .. i_end  {
                                for k in k_start .. k_end {
                                    for j in j_start .. j_end {
                                        out[i*cols+j] += 
                                            a_tile[(i - i_start)*tile+(k - k_start)] * b_tile[(k - k_start)*tile+(j - j_start)];
                                    }
                                }
                            }
                        }
                     
                    }
                }
            });
        }
    });

    Some(Matrix { data, rows, cols })
}


pub fn matmul_tiled_parallel3(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
        return None;
    }
    let tile = 256;
    let rows = m1.rows;
    let cols = m2.cols;
    let inner = m1.cols;

    let num_tile_rows = (rows + tile - 1) / tile;
    let num_threads = if num_tile_rows >= 8 {8} else {num_tile_rows};
    let high_tile_rows_per_thread = (num_tile_rows + num_threads - 1) / num_threads;
    let low_tile_rows_per_thread = high_tile_rows_per_thread - 1;
    let num_high = if num_tile_rows % num_threads == 0 {num_threads} else {num_threads};


    let mut data = vec![0.0; rows * cols];
    let data_ptr = data.as_mut_ptr() as usize;
    let m1_ptr = m1.data.as_ptr() as usize;
    let m2_ptr = m2.data.as_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..num_threads {
            let high = t + 1 <= num_high;
            let start_tile_row = if high {high_tile_rows_per_thread * t} else 
            {high_tile_rows_per_thread * num_high + low_tile_rows_per_thread * (t - num_high)};

            let local_tile_rows = if high {high_tile_rows_per_thread} else {low_tile_rows_per_thread};

            let end_tile_row = min(start_tile_row + local_tile_rows - 1, num_tile_rows - 1);
            let start_row = start_tile_row * tile;
            let end_row = min((end_tile_row + 1) * tile - 1, rows -1);
            s.spawn(move || {
               let out = unsafe { std::slice::from_raw_parts_mut(
                    (data_ptr as *mut f64).add(start_row *cols), 
                    (end_row - start_row +1) * cols
                )};

                let a = unsafe { std::slice::from_raw_parts(
                    (m1_ptr as *const f64).add(start_row * inner),
                     (end_row - start_row + 1) * inner
                )};

                let b = unsafe { std::slice::from_raw_parts(
                    m2_ptr as *const f64, inner * cols
                )};
            
                for i_t in 0 .. end_tile_row - start_tile_row + 1 {
                    let i_start = i_t * tile;
                    let i_end = min(i_start + tile, end_row - start_row + 1);
                    //println!("{}", i_end);
                    for k_t in 0 .. (inner + tile - 1) / tile {
                        let k_start = k_t * tile;
                        let k_end = min(k_start + tile, inner);

                        let mut a_tile = vec![0.0; tile * tile];

                        for i in i_start..i_end {
                            for k in k_start..k_end {
                                a_tile[(i-i_start)*tile+(k-k_start)] = a[i * inner + k];
                            }
                        }
                        
                        for j_t in 0..(cols + tile - 1) / tile {
                            let j_start = j_t * tile;
                            let j_end = min(j_start + tile, cols);
                            let j_size = j_end - j_start;
                            let i_size = i_end - i_start;

                            let mut b_tile = vec![0.0; tile * tile];
                            let mut c_tile = vec![0.0; tile * tile];

                            for k in k_start..k_end {
                                for j in j_start..j_end {
                                    b_tile[(k - k_start) * tile + (j - j_start)] = b[k * cols + j];
                                }
                            }

                            for i in 0..i_size {
                                for k in 0..k_end - k_start {
                                    for j in 0..j_size {
                                        c_tile[i * tile + j] +=
                                            a_tile[i * tile + k] * b_tile[k * tile + j];
                                    }
                                }
                            }

                            // flush c_tile to out
                            for i in 0..i_size {
                                for j in 0..j_size {
                                    out[(i_start + i) * cols + (j_start + j)] += c_tile[i * tile + j];
                                }
                            }
                        }
                     
                    }
                }
            });
        }
    });

    Some(Matrix { data, rows, cols })
}


pub fn matmul_parallel(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
        return None;
    }
    let num_threads = 8;
    let rows = m1.rows;
    let cols = m2.cols;
    let inner = m1.cols;
    let rows_per_thread = (rows + num_threads - 1) / num_threads;

    let mut data = vec![0.0; rows * cols];
    let data_ptr = data.as_mut_ptr() as usize;
    let m1_ptr = m1.data.as_ptr() as usize;
    let m2_ptr = m2.data.as_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..num_threads {
            let start_row = t * rows_per_thread;
            let end_row = min(start_row + rows_per_thread, rows);

            s.spawn(move || {
                let out = unsafe { std::slice::from_raw_parts_mut(
                    (data_ptr as *mut f64).add(start_row * cols), 
                    (end_row - start_row) * cols
                )};
                let a = unsafe { std::slice::from_raw_parts(
                    m1_ptr as *const f64, rows * inner
                )};
                let b = unsafe { std::slice::from_raw_parts(
                    m2_ptr as *const f64, inner * cols
                )};

                for i in start_row..end_row {
                    for k in 0..inner {
                        for j in 0..cols {
                            out[(i - start_row) * cols + j] += 
                                a[i * inner + k] * b[k * cols + j];
                        }
                    }
                }
            });
        }
    });

    Some(Matrix { data, rows, cols })
}


pub fn matmul_parallel_simple(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
    if m1.cols != m2.rows {
        return None;
    }
    
    let cols = m2.cols;
    let mut data = vec![0.0; m1.rows * cols];
    
    data.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, row)| {
            for k in 0..m1.cols {
                for j in 0..cols {
                    row[j] += m1[(i, k)] * m2[(k, j)];
                }
            }
        });
    
    Some(Matrix { data, rows: m1.rows, cols })
}

pub fn matmul_optimal_order(sizes: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let n = sizes.len();
    let mut m = vec![vec![0; n]; n];
    let mut s = vec![vec![0; n]; n];

    for l in 1 .. n  {
        for start in 0 .. n - l {
            let mut min = usize::MAX;
            let mut min_i : usize = 0;
            for i in start .. start + l {
                let cost = m[start][i] + m[i + 1][start + l] + sizes[start].0 * sizes[i].1 * sizes[start + l].1;
                if cost < min {
                    min = cost;
                    min_i = i;
                }
            }
            m[start][start + l] = min;
            s[start][start + l] = min_i;
        }
    }
    for i in 0..n {
    for j in 0..n {
        print!("{} ", s[i][j]);
    }
    println!();
}

    s
}

pub fn matmul_chained(matrices : &[Matrix]) -> Option<Matrix> {
    let mut sizes : Vec<(usize, usize)> = vec![];
    let mut latest_column_size : usize = matrices[0].rows;
    for matrix in matrices {
        if matrix.rows != latest_column_size {return None;}
        sizes.push((matrix.rows, matrix.cols));
        latest_column_size = matrix.cols;
    }
    let ord_mat = matmul_optimal_order(&sizes);
    Some(matmul_chained_rec(&matrices, &ord_mat, 0, matrices.len() - 1))
}

fn matmul_chained_rec(matrices : &[Matrix], ord_mat : &Vec<Vec<usize>>, i : usize, j : usize) -> Matrix {
    println!("multiplying {} to {}, split at {}", i, j, ord_mat[i][j]);
    if j == i + 1 {matmul(&matrices[i], &matrices[j]).unwrap()}
    else if j == i {
        println!("base case, returning matrix {}", i);
        matrices[i].clone()
    }
    else {
        let a = matmul_chained_rec(&matrices, &ord_mat, i, ord_mat[i][j]);
        let b = matmul_chained_rec(&matrices, &ord_mat, ord_mat[i][j] + 1, j);
        matmul(&a, &b).unwrap()

    }   
}
