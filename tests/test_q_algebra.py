import pytest
import numpy as np

from tsallis_stat.q_algebra import QAlgebra
from tsallis_stat.q_functions import exp_q, log_q


def test_q_algebra():
    q = 0.5
    a_v = 3
    b_v = 2
    a = QAlgebra(a_v, q)
    b = QAlgebra(b_v, q)

    assert a + b == QAlgebra(a_v + b_v, q)
    assert a - b == QAlgebra(a_v - b_v, q)

    assert exp_q(a+b) == (exp_q(a) * exp_q(b)), f"Multiplication is not correct for {a} and {b}"
    assert log_q(a*b) == (log_q(a) + log_q(b)), f"Division is not correct for {a} and {b}"

    x = QAlgebra(np.array([1, 2]), q)
    y = QAlgebra(np.array([3, 4]), q)
    assert x @ y == (x[0] * y[0] + x[1] * y[1]), f"Inner product is not correct for {x} and {y}"

    A = QAlgebra(np.array([[1, 2], [3, 4]]), q)
    B = QAlgebra(np.array([[5, 6], [7, 8]]), q)
 
    Ax = QAlgebra(np.array([
        (A[0, :] @ x).value,
        (A[1, :] @ x).value
    ]), q)
    xA = QAlgebra(np.array([
        (x @ A[:, 0]).value,
        (x @ A[:, 1]).value
    ]), q)
    AB = QAlgebra(np.array([
        (A @ B[:, 0]).value,
        (A @ B[:, 1]).value
    ]), q)
    assert (A @ x) == Ax, f"Matrix-vector multiplication is not correct for {A} and {x}"
    assert (x @ A) == xA, f"Vector-matrix multiplication is not correct for {x} and {A}"
    assert (A @ B) == AB, f"Matrix-matrix multiplication is not correct for {A} and {B}"

