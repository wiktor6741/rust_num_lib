mod matrix;

pub use self::matrix::Matrix;
pub use self::matrix::matmul_better;
pub use self::matrix::matmul_naive;
pub use self::matrix::matmul_benchmark;
pub use self::matrix::matmul_tiled_parallel;
pub use self::matrix::matrices_equal;
pub use self::matrix::random_matrix;
pub use self::matrix::matmul_tiled_parallel2;
pub use self::matrix::matmul_parallel_simple;