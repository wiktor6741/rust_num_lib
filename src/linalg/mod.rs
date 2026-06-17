mod matmul;
mod matrix;
mod benchmark;
mod solve;
mod eigenv;
mod cholesky;

pub use self::matrix::Matrix;
pub use self::matrix::MatrixError;
pub use self::matrix::matrices_equal;
pub use self::matrix::apply_along_axis;
pub use self::matrix::assert_matrices_equal;
pub use self::matrix::vector;

pub use self::matmul::matmul_better;
pub use self::matmul::matmul_naive;
pub use self::matmul::matmul_tiled_parallel;
pub use self::matmul::matmul_parallel;
pub use self::matmul::matmul;
pub use self::matmul::matmul_chained;



pub use self::solve::gauss_elimination;
pub use self::solve::doolitle_factorize;
pub use self::solve::doolitle_solve;
pub use self::solve::solve_multiple;

pub use self::eigenv::qr_factorize;

pub use self::cholesky::cholesky;

pub use self::benchmark::matmul_benchmark;
pub use self::benchmark::chain_benchmark;
pub use self::benchmark::eigen_benchmark;
pub use self::benchmark::qr_steps_benchmark;
pub use self::benchmark::eigen_full_benchmark;
pub use self::benchmark::solve_benchmark;
pub use self::benchmark::solve_multiple_rhs_benchmark;


