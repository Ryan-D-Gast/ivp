"""Test edge cases and special scenarios."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal, assert_no_warnings
from ivp import solve_ivp
from test_helpers import fun_zero


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_no_integration(method):
    """Test when t_span start equals end."""
    sol = solve_ivp(lambda t, y: -y, [4, 4], [2, 3],
                    method=method, dense_output=True)
    assert_equal(sol.sol(4), [2, 3])
    assert_equal(sol.sol([4, 5, 6]), [[2, 2, 2], [3, 3, 3]])


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_empty(method):
    """Test with empty state vector."""
    def fun(t, y):
        return np.zeros((0,))

    y0 = np.zeros((0,))
    sol = assert_no_warnings(solve_ivp, fun, [0, 10], y0,
                             method=method, dense_output=True)
    assert_equal(sol.sol(10), np.zeros((0,)))
    assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_integration_zero_rhs(method):
    """Test integration with zero right-hand side."""
    result = solve_ivp(fun_zero, [0, 10], np.ones(3), method=method)
    assert_(result.success)
    assert_equal(result.status, 0)
    assert_allclose(result.y, 1.0, rtol=1e-15)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_zero_interval(method):
    """Test when t0 equals tf."""
    def f(t, y):
        return 2 * y
    res = solve_ivp(f, (0.0, 0.0), np.array([1.0]), method=method)
    assert res.success
    assert_allclose(res.y[0, -1], 1.0)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_tbound_respected_small_interval(method):
    """Regression test for gh-17341 - small interval boundary."""
    SMALL = 1e-4

    def f(t, y):
        if t > SMALL:
            raise ValueError("Function was evaluated outside interval")
        return 2 * y
    res = solve_ivp(f, (0.0, SMALL), np.array([1]), method=method)
    assert res.success


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_tbound_respected_larger_interval(method):
    """Regression test for gh-8848 - larger interval boundary."""
    def V(r):
        return -11/r + 10 * r / (0.05 + r**2)

    def func(t, p):
        if t < -17 or t > 2:
            raise ValueError("Function was evaluated outside interval")
        P = p[0]
        Q = p[1]
        r = np.exp(t)
        dPdr = r * Q
        dQdr = -2.0 * r * ((-0.2 - V(r)) * P + 1 / r * Q)
        return np.array([dPdr, dQdr])

    result = solve_ivp(func,
                       (-17, 2),
                       y0=np.array([1, -11]),
                       max_step=0.03,
                       vectorized=False,
                       t_eval=None,
                       atol=1e-8,
                       rtol=1e-5,
                       method=method)
    assert result.success


@pytest.mark.timeout(10)
@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF'])
def test_tbound_respected_oscillator(method):
    """Regression test for gh-9198 - oscillator boundary."""
    def reactions_func(t, y):
        if (t > 205):
            raise ValueError("Called outside interval")
        yprime = np.array([1.73307544e-02,
                           6.49376470e-06,
                           0.00000000e+00,
                           0.00000000e+00])
        return yprime

    def run_sim2(t_end, n_timepoints=10, shortest_delay_line=10000000):
        init_state = np.array([134.08298555, 138.82348612, 100., 0.])
        t0 = 100.0
        t1 = 200.0
        return solve_ivp(reactions_func,
                         (t0, t1),
                         init_state.copy(),
                         dense_output=True,
                         max_step=t1 - t0,
                         method=method)
    result = run_sim2(1000, 100, 100)
    assert result.success
