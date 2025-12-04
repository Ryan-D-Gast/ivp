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


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_max_steps_parameter(method):
    """Test that max_steps parameter is correctly passed to the solver."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    # Use max_steps=1 to guarantee triggering the limit
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method=method, max_steps=1)
    
    # Should fail because 1 step is too few
    assert_(not res.success)
    assert_equal(res.status, -1)
    assert_('NeedLargerNMax' in res.message or 'max' in res.message.lower())


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_max_steps_large_value(method):
    """Test that large max_steps allows completion."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    # Use a large max_steps value
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method=method, max_steps=1_000_000)
    
    # Should succeed
    assert_(res.success)
    assert_equal(res.status, 0)


@pytest.mark.timeout(120)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_default_max_steps_is_unlimited(method):
    """Test that default max_steps is effectively unlimited (like scipy's np.inf).
    
    This test uses a simple ODE over a very long integration span with loose 
    tolerances. This requires many steps but should complete if max_steps is 
    unlimited.
    """
    # Simple exponential decay - not stiff, but we'll use a very long span
    # with small tolerances to force many steps
    def exponential_decay(t, y):
        return [-0.001 * y[0]]  # Very slow decay
    
    y0 = [1.0]
    # Very long time span 
    t_span = [0, 100000]
    
    # Tight tolerances to force more steps
    rtol = 1e-8
    atol = 1e-10
    
    # Don't pass max_steps - should use unlimited (not solver default of 100,000)
    res = solve_ivp(exponential_decay, t_span, y0, method=method, rtol=rtol, atol=atol)
    
    # The integration should succeed without hitting any max_steps limit
    # If max_steps defaults were being applied incorrectly, this would fail
    # with NeedLargerNMax
    assert_(res.success, f"Method {method} failed with message: {res.message}")
    assert_equal(res.status, 0)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['Radau', 'BDF'])
def test_min_step_parameter(method):
    """Test that min_step parameter is correctly passed to stiff solvers."""
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    
    # Use min_step - should still work with a small value
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                    atol=atol, method=method, min_step=1e-10)
    
    # Should succeed
    assert_(res.success, f"Method {method} failed with message: {res.message}")
    assert_equal(res.status, 0)
