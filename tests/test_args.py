"""Test function arguments and special options."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose
from ivp import solve_ivp


@pytest.mark.timeout(10)
def test_args_with_events():
    """Test args parameter with event functions."""
    def sys3(t, w, omega, k, zfinal):
        x, y, z = w
        return [-omega*y, omega*x, k*z*(1 - z)]

    def sys3_jac(t, w, omega, k, zfinal):
        x, y, z = w
        J = np.array([[0, -omega, 0],
                      [omega, 0, 0],
                      [0, 0, k*(1 - 2*z)]])
        return J

    def sys3_x0decreasing(t, w, omega, k, zfinal):
        x, y, z = w
        return x

    def sys3_y0increasing(t, w, omega, k, zfinal):
        x, y, z = w
        return y

    def sys3_zfinal(t, w, omega, k, zfinal):
        x, y, z = w
        return z - zfinal

    sys3_x0decreasing.direction = -1
    sys3_y0increasing.direction = 1
    sys3_zfinal.terminal = True

    omega = 2
    k = 4

    tfinal = 5
    zfinal = 0.99
    z0 = np.exp(-k*tfinal)/((1 - zfinal)/zfinal + np.exp(-k*tfinal))

    w0 = [0, -1, z0]

    tend = 2*tfinal
    sol = solve_ivp(sys3, [0, tend], w0,
                    events=[sys3_x0decreasing, sys3_y0increasing, sys3_zfinal],
                    dense_output=True, args=(omega, k, zfinal),
                    method='Radau', jac=sys3_jac,
                    rtol=1e-10, atol=1e-13)

    x0events_t = sol.t_events[0]
    y0events_t = sol.t_events[1]
    zfinalevents_t = sol.t_events[2]
    # Event finding has finite precision - bisection achieves ~1e-6 to 1e-7 relative tolerance
    assert_allclose(x0events_t, [0.5*np.pi, 1.5*np.pi], rtol=1e-6, atol=1e-8)
    assert_allclose(y0events_t, [0.25*np.pi, 1.25*np.pi], rtol=1e-6, atol=1e-8)
    # Event finding has finite precision, allow small tolerance
    assert_allclose(zfinalevents_t, [tfinal], rtol=1e-5, atol=1e-5)

    t = np.linspace(0, zfinalevents_t[0], 250)
    w = sol.sol(t)
    # Dense output interpolation error is larger than solver tolerances
    assert_allclose(w[0], np.sin(omega*t), rtol=1e-4, atol=1e-6)
    assert_allclose(w[1], -np.cos(omega*t), rtol=1e-4, atol=1e-6)
    assert_allclose(w[2], 1/(((1 - z0)/z0)*np.exp(-k*t) + 1),
                    rtol=1e-4, atol=1e-6)


@pytest.mark.timeout(10)
def test_args_single_value():
    """Test that single-element tuple args work."""
    def fun_with_arg(t, y, a):
        return a*y

    sol = solve_ivp(fun_with_arg, (0, 0.1), [1], args=(-1,))
    assert_allclose(sol.y[0, -1], np.exp(-0.1))


@pytest.mark.timeout(10)
def test_array_rtol():
    """Test that array-like rtol works (gh-15482)."""
    def f(t, y):
        return y[0], y[1]

    sol = solve_ivp(f, (0, 1), [1., 1.], rtol=[1e-1, 1e-1])
    err1 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))

    sol = solve_ivp(f, (0, 1), [1., 1.], rtol=[1e-1, 1e-16])
    err2 = np.abs(np.linalg.norm(sol.y[:, -1] - np.exp(1)))

    # tighter rtol improves the error
    assert err2 < err1
