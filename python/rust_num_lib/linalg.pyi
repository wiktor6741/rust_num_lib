"""Linear algebra module: matrices, factorizations, eigenvalues, solving systems."""
from typing import Optional


class Matrix:
    """A dense row-major matrix of f64 values."""

    def __init__(self, rows: int, cols: int) -> None:
        """Create a new zero-filled matrix of shape (rows, cols)."""
        ...

    @staticmethod
    def zeros(rows: int, cols: int) -> "Matrix":
        """Create a (rows x cols) matrix filled with zeros."""
        ...

    @staticmethod
    def identity(size: int) -> "Matrix":
        """Create a (size x size) identity matrix."""
        ...

    @staticmethod
    def random(rows: int, cols: int, min: float, max: float) -> "Matrix":
        """Create a (rows x cols) matrix with values drawn uniformly from [min, max)."""
        ...

    @staticmethod
    def from_data(data: list[float], rows: int, cols: int) -> "Matrix":
        """Create a matrix from a flat row-major list. len(data) must equal rows * cols."""
        ...

    @property
    def rows(self) -> int:
        """Number of rows."""
        ...

    @property
    def cols(self) -> int:
        """Number of columns."""
        ...

    def transpose(self) -> "Matrix":
        """Return the transpose as a new matrix."""
        ...

    def determinant(self) -> float:
        """Determinant of a square matrix (via LU factorization)."""
        ...

    def trace(self) -> float:
        """Sum of the diagonal elements."""
        ...

    def norm(self) -> float:
        """Frobenius norm: sqrt of the sum of squared elements."""
        ...

    def rank(self) -> int:
        """Rank: number of non-zero pivots from LU factorization."""
        ...

    def eigenvalues(self, steps: Optional[int] = None) -> list[float]:
        """Eigenvalues, approximated with the QR algorithm.

        :param steps: number of QR iterations (default: built-in value).
        :return: list of eigenvalues.
        """
        ...

    def eigenvectors(self, steps: Optional[int] = None) -> tuple[list[float], list["Matrix"]]:
        """Eigenvalues and eigenvectors.

        :param steps: number of QR iterations.
        :return: tuple (eigenvalues, eigenvectors) where each eigenvector is a column Matrix.
        """
        ...

    def row(self, i: int) -> "Matrix":
        """Return row i as a (1 x cols) matrix."""
        ...

    def col(self, i: int) -> "Matrix":
        """Return column i as a (rows x 1) matrix."""
        ...

    def slice(
        self,
        row_range: Optional[tuple[int, int]] = None,
        row_step: Optional[int] = None,
        col_range: Optional[tuple[int, int]] = None,
        col_step: Optional[int] = None,
    ) -> "Matrix":
        """Extract a sub-matrix with optional (start, end) ranges and steps per axis.

        Negative steps reverse the axis. None selects the full axis.
        """
        ...

    def apply_along_axis(self, f: "callable", axis: int) -> "Matrix":
        """Apply f to each row or column, reducing it to a single value.

        :param f: callable taking a Matrix (row/column) and returning a float.
        :param axis: 0 applies f to each column (result 1 x cols),
                     1 applies f to each row (result rows x 1).
        """
        ...

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Matrix multiplication (the @ operator)."""
        ...

    def __add__(self, other: "Matrix") -> "Matrix":
        """Element-wise matrix addition."""
        ...

    def __mul__(self, scalar: float) -> "Matrix":
        """Scalar multiplication (matrix * scalar)."""
        ...

    def __rmul__(self, scalar: float) -> "Matrix":
        """Scalar multiplication (scalar * matrix)."""
        ...

    def __getitem__(self, idx: tuple[int, int]) -> float:
        """Element access: m[i, j]."""
        ...

    def __setitem__(self, idx: tuple[int, int], value: float) -> None:
        """Element assignment: m[i, j] = value."""
        ...


def matrix(data: list[list[float]]) -> Matrix:
    """Build a matrix from a 2D list, inferring dimensions from its shape.

    All rows must have the same length.
    """
    ...


def vector(data: list[float]) -> Matrix:
    """Build a column vector (n x 1 matrix) from a flat list."""
    ...


def matmul_chained(matrices: list[Matrix]) -> Matrix:
    """Multiply a chain of matrices using an optimal multiplication order."""
    ...


def solve(a: Matrix, b: Matrix) -> Matrix:
    """Solve the linear system A x = b for x."""
    ...


def solve_multiple(a: Matrix, bs: list[Matrix]) -> list[Matrix]:
    """Solve A x = b for multiple right-hand sides, factorizing A once."""
    ...


def qr_factorize(a: Matrix) -> tuple[Matrix, Matrix]:
    """QR factorization. Returns (Q, R) with Q orthonormal, R upper triangular, A = Q R."""
    ...


def lu_factorize(a: Matrix) -> tuple[Matrix, Matrix, list[int]]:
    """LU factorization with partial pivoting. Returns (L, U, perm) such that P A = L U."""
    ...


def cholesky(a: Matrix) -> Matrix:
    """Cholesky factorization of a symmetric positive-definite matrix.

    Returns lower-triangular L such that A = L Lᵀ.
    """
    ...
