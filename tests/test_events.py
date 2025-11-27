"""Test event handling functionality."""
import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from ivp import solve_ivp
from test_helpers import fun_rational, sol_rational, compute_error


@pytest.mark.timeout(10)
def test_events_RK23():
    """Test event detection with RK23 method."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method='RK23',
                    events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 1)
    assert_equal(len(res.t_events[1]), 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)


@pytest.mark.timeout(10)
def test_events_RK45():
    """Test event detection with RK45 method."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method='RK45',
                    events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 1)
    assert_equal(len(res.t_events[1]), 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)


@pytest.mark.timeout(10)
def test_events_DOP853():
    """Test event detection with DOP853 method."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method='DOP853',
                    events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 1)
    assert_equal(len(res.t_events[1]), 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)


@pytest.mark.timeout(10)
def test_events_Radau():
    """Test event detection with Radau method."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method='Radau',
                    events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 1)
    assert_equal(len(res.t_events[1]), 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)


@pytest.mark.timeout(10)
def test_events_BDF():
    """Test event detection with BDF method."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method='BDF',
                    events=(event_rational_1, event_rational_2))
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 1)
    assert_equal(len(res.t_events[1]), 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)
    assert_(7.3 < res.t_events[1][0] < 7.7)


@pytest.mark.timeout(10)
def test_terminal_event():
    """Test terminal event stops integration."""
    def event_rational_3(t, y):
        return t - 7.4

    event_rational_3.terminal = True

    res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method='RK45',
                    events=event_rational_3, dense_output=True)
    assert_equal(res.status, 1)
    assert_equal(len(res.t_events[0]), 1)
    assert_(7.3 < res.t_events[0][0] < 7.5)


@pytest.mark.timeout(10)
def test_event_direction_positive():
    """Test event with positive direction."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    event_rational_1.direction = 1

    res = solve_ivp(fun_rational, [5, 8], [1 / 3, 2 / 9], method='RK45',
                    events=event_rational_1)
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 1)
    assert_(5.3 < res.t_events[0][0] < 5.7)


@pytest.mark.timeout(10)
def test_event_direction_negative():
    """Test event with negative direction."""
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    event_rational_1.direction = -1

    res = solve_ivp(fun_rational, [5, 8], [1 / 3, 2 / 9], method='RK45',
                    events=event_rational_1)
    assert_equal(res.status, 0)
    assert_equal(len(res.t_events[0]), 0)


@pytest.mark.timeout(10)
def test_duplicate_timestamps():
    """Test that duplicate timestamps don't cause issues."""
    def upward_cannon(t, y):
        return [y[1], -9.80665]

    def hit_ground(t, y):
        return y[0]

    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(upward_cannon, [0, np.inf], [0, 0.01],
                    max_step=0.05 * 0.001 / 9.80665,
                    events=hit_ground, dense_output=True)
    assert_allclose(sol.sol(0.01), np.asarray([-0.00039033, -0.08806632]),
                    rtol=1e-5, atol=1e-8)
    assert_allclose(sol.t_events[0], np.asarray([0.00203943]), rtol=1e-5, atol=1e-8)
    assert sol.success
    assert_equal(sol.status, 1)
