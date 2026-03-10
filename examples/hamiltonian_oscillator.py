"""Hamiltonian example: harmonic oscillator in canonical coordinates."""

import ivp
import numpy as np


def position_derivative(t, p):
    return np.array([p[0]], dtype=float)


def momentum_derivative(t, q):
    return np.array([-q[0]], dtype=float)


def energy(state):
    q, p = state
    return 0.5 * (q * q + p * p)


t_eval = np.linspace(0.0, 20.0, 11)
sol = ivp.solve_ivp(
    (position_derivative, momentum_derivative),
    (0.0, 20.0),
    [1.0, 0.0],
    method="Yoshida4",
    step_size=0.05,
    t_eval=t_eval,
)

h0 = energy(sol.y[:, 0])
print(f"status: {sol.message}")
for t, q, p in zip(sol.t, sol.y[0], sol.y[1]):
    print(f"t = {t:>5.2f}, q = {q:>9.6f}, p = {p:>9.6f}, dH = {energy((q, p)) - h0:>9.2e}")
