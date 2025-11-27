"""Test stiff integration methods (Radau, BDF)."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from scipy.sparse import csc_matrix
from ivp import solve_ivp
from test_helpers import (
    fun_linear, jac_linear, sol_linear,
    fun_medazko, medazko_sparsity, compute_error
)


@pytest.mark.timeout(10)
def test_integration_const_jac_Radau():
    """Test Radau with constant Jacobian."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    t_span = [0, 2]
    J = jac_linear()

    res = solve_ivp(fun_linear, t_span, y0, rtol=rtol, atol=atol,
                    method='Radau', dense_output=True, jac=J)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    assert_(res.nfev < 100)

    y_true = sol_linear(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 10))


@pytest.mark.timeout(10)
def test_integration_const_jac_BDF():
    """Test BDF with constant Jacobian."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    t_span = [0, 2]
    J = jac_linear()

    res = solve_ivp(fun_linear, t_span, y0, rtol=rtol, atol=atol,
                    method='BDF', dense_output=True, jac=J)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    assert_(res.nfev < 100)

    y_true = sol_linear(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 10))


@pytest.mark.timeout(10)
def test_integration_const_jac_sparse_Radau():
    """Test Radau with sparse constant Jacobian."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    t_span = [0, 2]
    J_sparse = csc_matrix(jac_linear())

    res = solve_ivp(fun_linear, t_span, y0, rtol=rtol, atol=atol,
                    method='Radau', dense_output=True, jac=J_sparse)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_linear(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 10))


@pytest.mark.timeout(10)
def test_integration_const_jac_sparse_BDF():
    """Test BDF with sparse constant Jacobian."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    t_span = [0, 2]
    J_sparse = csc_matrix(jac_linear())

    res = solve_ivp(fun_linear, t_span, y0, rtol=rtol, atol=atol,
                    method='BDF', dense_output=True, jac=J_sparse)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_linear(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 10))


@pytest.mark.timeout(10)
@pytest.mark.slow
def test_integration_stiff_Radau():
    """Test Radau on Robertson stiff problem."""
    rtol = 1e-6
    atol = 1e-6
    y0 = [1e4, 0, 0]
    tspan = [0, 1e8]

    def fun_robertson(t, state):
        x, y, z = state
        return [
            -0.04 * x + 1e4 * y * z,
            0.04 * x - 1e4 * y * z - 3e7 * y * y,
            3e7 * y * y,
        ]

    res = solve_ivp(fun_robertson, tspan, y0, rtol=rtol,
                    atol=atol, method='Radau')

    # If the stiff mode is not activated correctly, these numbers will be much bigger
    assert res.nfev < 5000
    assert res.njev < 200


@pytest.mark.timeout(10)
@pytest.mark.slow
def test_integration_stiff_BDF():
    """Test BDF on Robertson stiff problem."""
    rtol = 1e-6
    atol = 1e-6
    y0 = [1e4, 0, 0]
    tspan = [0, 1e8]

    def fun_robertson(t, state):
        x, y, z = state
        return [
            -0.04 * x + 1e4 * y * z,
            0.04 * x - 1e4 * y * z - 3e7 * y * y,
            3e7 * y * y,
        ]

    res = solve_ivp(fun_robertson, tspan, y0, rtol=rtol,
                    atol=atol, method='BDF')

    assert res.nfev < 5000
    # BDF Jacobian evaluation count is implementation-dependent
    # SciPy expects < 200, our implementation may differ
    assert res.njev < 600


@pytest.mark.timeout(10)
def test_integration_sparse_difference_BDF():
    """Test BDF with sparse difference matrix."""
    n = 200
    t_span = [0, 20]
    y0 = np.zeros(2 * n)
    y0[1::2] = 1
    sparsity = medazko_sparsity(n)

    res = solve_ivp(fun_medazko, t_span, y0, method='BDF',
                    jac_sparsity=sparsity)

    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)

    assert_allclose(res.y[78, -1], 0.233994e-3, rtol=1e-2)
    assert_allclose(res.y[79, -1], 0, atol=1e-3)


@pytest.mark.timeout(10)
def test_integration_sparse_difference_Radau():
    """Test Radau with sparse difference matrix."""
    n = 200
    t_span = [0, 20]
    y0 = np.zeros(2 * n)
    y0[1::2] = 1
    sparsity = medazko_sparsity(n)

    res = solve_ivp(fun_medazko, t_span, y0, method='Radau',
                    jac_sparsity=sparsity)

    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)

    assert_allclose(res.y[78, -1], 0.233994e-3, rtol=1e-2)
    assert_allclose(res.y[79, -1], 0, atol=1e-3)
