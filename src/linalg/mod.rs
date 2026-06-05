mod matmul;
mod matrix;
mod benchmark;

pub use self::matrix::Matrix;
pub use self::matrix::matrices_equal;
pub use self::matrix::random_matrix;

pub use self::matmul::matmul_better;
pub use self::matmul::matmul_naive;
pub use self::matmul::matmul_benchmark;
pub use self::matmul::matmul_tiled_parallel;
pub use self::matmul::matmul_parallel;
pub use self::matmul::matmul;
pub use self::matmul::matmul_tiled_parallel3;
pub use self::matmul::matmul_parallel_simple;
pub use self::matmul::matmul_optimal_order;
pub use self::matmul::matmul_chained;
pub use self::matmul::chain_benchmark;