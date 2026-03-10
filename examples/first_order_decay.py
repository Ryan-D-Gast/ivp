"""First-order example: exponential decay with an analytical reference."""

import math

import ivp
import numpy as np


def decay(t, y):
    return np.array([-0.5 * y[0]], dtype=float)


t_eval = np.linspace(0.0, 10.0, 11)
sol = ivp.solve_ivp(
    decay,
    (0.0, 10.0),
    [1.0],
    method="RK45",
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)

print(f"status: {sol.message}")
for t, y in zip(sol.t, sol.y[0]):
    exact = math.exp(-0.5 * t)
    print(f"t = {t:>4.1f}, y = {y:>10.7f}, exact = {exact:>10.7f}, err = {abs(y - exact):.2e}")
