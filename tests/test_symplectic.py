"""Tests for the structured symplectic Python API."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ivp import solve_ivp


def acceleration(t, q):
    return np.array([-q[0]])


def position_derivative(t, p):
    return np.array([p[0]])


def momentum_derivative(t, q):
    return np.array([-q[0]])


class LegacyHarmonicHamiltonian:
    def drift(self, t, p):
        return np.array([p[0]])

    def kick(self, t, q):
        return np.array([-q[0]])


def test_symplectic_solve_ivp_second_order_plain_callable_dense_output():
    sol = solve_ivp(
        acceleration,
        (0.0, 2.0),
        [1.0, 0.0],
        method="VelocityVerlet",
        step_size=0.05,
        dense_output=True,
    )

    assert sol.success
    assert sol.sol is not None
    y_mid = sol.sol(0.5)
    assert_allclose(y_mid, [np.cos(0.5), -np.sin(0.5)], atol=2e-3, rtol=0.0)


def test_symplectic_solve_ivp_hamiltonian_tuple_t_eval():
    t_eval = np.array([0.0, 0.1, 0.7, 1.3, 2.0])
    sol = solve_ivp(
        (position_derivative, momentum_derivative),
        (0.0, 2.0),
        [1.0, 0.0],
        method="Yoshida4",
        step_size=0.2,
        t_eval=t_eval,
    )

    assert sol.success
    assert_allclose(sol.t, t_eval, atol=1e-12, rtol=0.0)
    assert sol.y.shape == (2, len(t_eval))


def test_symplectic_solve_ivp_hamiltonian_keyword_callbacks():
    sol = solve_ivp(
        None,
        (0.0, 2.0),
        [1.0, 0.0],
        method="Yoshida4",
        step_size=0.1,
        position_derivative=position_derivative,
        momentum_derivative=momentum_derivative,
    )

    assert sol.success
    assert_allclose(sol.y[:, -1], [np.cos(2.0), -np.sin(2.0)], atol=2e-4, rtol=0.0)


def test_symplectic_solve_ivp_legacy_object_callbacks_still_work():
    sol = solve_ivp(
        LegacyHarmonicHamiltonian(),
        (0.0, 2.0),
        [1.0, 0.0],
        method="Yoshida4",
        step_size=0.1,
    )

    assert sol.success
    assert_allclose(sol.y[:, -1], [np.cos(2.0), -np.sin(2.0)], atol=2e-4, rtol=0.0)


def test_symplectic_second_order_wrong_output_length_raises_python_error():
    def bad_acceleration(t, q):
        return np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="returned 2 values, but 1 were expected"):
        solve_ivp(
            bad_acceleration,
            (0.0, 1.0),
            [1.0, 0.0],
            method="VelocityVerlet",
            step_size=0.1,
        )


def test_symplectic_hamiltonian_missing_pair_raises_type_error():
    with pytest.raises(
        TypeError,
        match="must either both be provided or both be omitted",
    ):
        solve_ivp(
            None,
            (0.0, 1.0),
            [1.0, 0.0],
            method="Yoshida4",
            step_size=0.1,
            position_derivative=position_derivative,
        )


def test_symplectic_callback_exception_becomes_python_runtime_error():
    def failing_acceleration(t, q):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="acceleration callback raised an exception"):
        solve_ivp(
            failing_acceleration,
            (0.0, 1.0),
            [1.0, 0.0],
            method="VelocityVerlet",
            step_size=0.1,
        )
