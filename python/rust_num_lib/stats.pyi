"""Statistics module: descriptive statistics and sampling from distributions."""
from .linalg import Matrix


def mean(m: Matrix) -> float:
    """Arithmetic mean of all elements in the matrix."""
    ...


def median(m: Matrix) -> float:
    """Median of all elements in the matrix."""
    ...


def std_dev(m: Matrix) -> float:
    """Standard deviation of all elements in the matrix."""
    ...


def variance(m: Matrix) -> float:
    """Variance of all elements in the matrix."""
    ...


def normal_sample(n: int, mu: float, sigma: float) -> Matrix:
    """Draw n samples from a normal distribution N(mu, sigma^2).

    :return: an (n x 1) column vector of samples.
    """
    ...


def multivariate_normal_sample(n_samples: int, mu: Matrix, sigma: Matrix) -> Matrix:
    """Draw samples from a multivariate normal distribution N(mu, Sigma).

    :param mu: mean vector.
    :param sigma: covariance matrix (symmetric positive-definite).
    :return: an (n_samples x dim) matrix where each row is one sample.
    """
    ...
