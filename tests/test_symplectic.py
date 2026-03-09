"""Tests for the structured symplectic Python API."""

import numpy as np
from numpy.testing import assert_allclose

from ivp import solve_ivp


class HarmonicSecondOrder:
    def acceleration(self, t, q):
        return np.array([-q[0]])


class HarmonicHamiltonian:
    def drift(self, t, p):
        return np.array([p[0]])

    def kick(self, t, q):
        return np.array([-q[0]])


def test_symplectic_solve_ivp_second_order_dense_output():
    sol = solve_ivp(
        HarmonicSecondOrder(),
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


def test_symplectic_solve_ivp_hamiltonian_t_eval():
    t_eval = np.array([0.0, 0.1, 0.7, 1.3, 2.0])
    sol = solve_ivp(
        HarmonicHamiltonian(),
        (0.0, 2.0),
        [1.0, 0.0],
        method="Yoshida4",
        step_size=0.2,
        t_eval=t_eval,
    )

    assert sol.success
    assert_allclose(sol.t, t_eval, atol=1e-12, rtol=0.0)
    assert sol.y.shape == (2, len(t_eval))
