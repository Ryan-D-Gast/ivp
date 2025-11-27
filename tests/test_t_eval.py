"""Test t_eval parameter functionality."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from ivp import solve_ivp
from test_helpers import fun_rational, sol_rational, jac_rational, compute_error


@pytest.mark.timeout(10)
def test_t_eval_forward():
    """Test t_eval in forward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_t_eval_backward():
    """Test t_eval in backward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 1]
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
def test_t_eval_irregular_forward():
    """Test irregular t_eval points in forward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    
    t_eval = [5, 5.01, 7, 8, 8.01, 9]
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_t_eval_irregular_backward():
    """Test irregular t_eval points in backward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    
    t_eval = [5, 4.99, 3, 1.5, 1.1, 1.01, 1]
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
def test_t_eval_interior_forward():
    """Test t_eval with interior points only in forward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    
    t_eval = [5.01, 7, 8, 8.01]
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


@pytest.mark.timeout(10)
def test_t_eval_interior_backward():
    """Test t_eval with interior points only in backward direction."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    
    t_eval = [4.99, 3, 1.5, 1.1, 1.01]
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
def test_t_eval_dense_output():
    """Test t_eval with dense output."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    res_d = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                      t_eval=t_eval, dense_output=True)
    
    assert_equal(res.t, t_eval)
    assert_(res.success)
    assert_equal(res.status, 0)

    assert_equal(res.t, res_d.t)
    assert_equal(res.y, res_d.y)
    assert_(res_d.success)
    assert_equal(res_d.status, 0)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_t_eval_early_event(method):
    """Test t_eval with early terminal event."""
    def early_event(t, y):
        return t - 7

    early_event.terminal = True

    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    t_eval = np.linspace(7.5, 9, 16)
    
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    method=method, t_eval=t_eval, events=early_event,
                    jac=jac_rational)
    assert res.success
    assert res.status == 1
    assert len(res.t_events) == 1
    assert res.t_events[0].size == 1
    # Event time should be very close to 7 (within root-finding tolerance)
    assert_allclose(res.t_events[0][0], 7, rtol=1e-10, atol=1e-10)
