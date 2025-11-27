"""Test step control options (max_step, first_step)."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from ivp import solve_ivp
from test_helpers import fun_rational, sol_rational, compute_error


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_max_step_forward(method):
    """Test max_step parameter in forward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    max_step=0.5, atol=atol, method=method,
                    dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_equal(res.t[-1], t_span[-1])
    assert_(np.all(np.abs(np.diff(res.t)) <= 0.5 + 1e-15))
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_max_step_backward(method):
    """Test max_step parameter in backward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 1]
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    max_step=0.5, atol=atol, method=method,
                    dense_output=True)
    assert_equal(res.t[0], t_span[0])
    assert_equal(res.t[-1], t_span[-1])
    assert_(np.all(np.abs(np.diff(res.t)) <= 0.5 + 1e-15))
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_first_step_forward(method):
    """Test first_step parameter in forward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    first_step = 0.1
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    max_step=0.5, atol=atol, method=method,
                    dense_output=True, first_step=first_step)

    assert_equal(res.t[0], t_span[0])
    assert_equal(res.t[-1], t_span[-1])
    assert_allclose(first_step, np.abs(res.t[1] - 5))
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_first_step_backward(method):
    """Test first_step parameter in backward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 1]
    first_step = 0.1
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    max_step=0.5, atol=atol, method=method,
                    dense_output=True, first_step=first_step)

    assert_equal(res.t[0], t_span[0])
    assert_equal(res.t[-1], t_span[-1])
    assert_allclose(first_step, np.abs(res.t[1] - 5))
    assert_(res.success)
    assert_equal(res.status, 0)
