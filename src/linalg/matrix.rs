use std::cmp::min_by;
use std::ops::{Index, IndexMut};
use std::time::Instant;
use rand::Rng;
use std::cmp::min;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows:  usize,
    pub cols: usize,
}


impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        &mut self.data[row * self.cols + col]
    }
}

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


pub fn matmul_tiled_parallel2(m1: &Matrix, m2: &Matrix) -> Option<Matrix> {
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
            println!("{}", start_row);
            let end_row = min((end_tile_row + 1) * tile - 1, rows -1);
            println!("{}", end_row);
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
            //println!("{}", start_row);
            let end_row = min((end_tile_row + 1) * tile - 1, rows -1);
            //println!("{}", end_row);
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



pub fn random_matrix(rows: usize, cols: usize, min : f64, max : f64) -> Matrix {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (0..rows*cols).map(|_| rng.gen_range(min..max)).collect();
    Matrix{data: v, rows, cols}
}

pub fn matrices_equal(m1: &Matrix, m2: &Matrix, epsilon: f64) -> bool {
    if m1.rows != m2.rows || m1.cols != m2.cols {
        return false;
    }
    
    for i in 0..m1.rows {
        for j in 0..m1.cols {
            if (m1[(i,j)] - m2[(i,j)]).abs() > epsilon {
                return false;
            }
        }
    }
    
    true
}