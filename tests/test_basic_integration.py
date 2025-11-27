"""Test basic integration functionality."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from ivp import solve_ivp
from test_helpers import (
    fun_rational, fun_rational_vectorized, jac_rational, jac_rational_sparse,
    sol_rational, compute_error
)


@pytest.mark.timeout(10)
def test_integration_RK23_forward():
    """Test RK23 method forward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='RK23', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_integration_RK45_forward():
    """Test RK45 method forward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='RK45', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_integration_DOP853_forward():
    """Test DOP853 method forward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='DOP853', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_integration_Radau_forward():
    """Test Radau method forward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='Radau', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_integration_BDF_forward():
    """Test BDF method forward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='BDF', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_integration_RK23_backward():
    """Test RK23 method backward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 1]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='RK23', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
def test_integration_RK45_backward():
    """Test RK45 method backward integration."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 1]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method='RK45', dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
def test_integration_vectorized():
    """Test vectorized function evaluation."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational_vectorized, t_span, y0, rtol=rtol,
                    atol=atol, method='RK45', dense_output=True,
                    vectorized=True)
    assert_equal(res.t[0], t_span[0])
    assert_(res.success)
    assert_equal(res.status, 0)
    
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))
