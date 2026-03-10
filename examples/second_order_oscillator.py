"""Second-order example: 3D harmonic oscillator with Velocity Verlet."""

import ivp
import numpy as np


def acceleration(t, q):
    return -np.asarray(q, dtype=float)


def energy(state):
    q = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:], dtype=float)
    return 0.5 * (np.dot(q, q) + np.dot(v, v))


t_eval = np.linspace(0.0, 20.0, 11)
y0 = [
    1.0, 0.0, -0.5,   # q0 = [x0, y0, z0]
    0.0, 1.0, 0.25,   # v0 = [vx0, vy0, vz0]
]

sol = ivp.solve_ivp(
    acceleration,
    (0.0, 20.0),
    y0,
    method="VelocityVerlet",
    step_size=0.05,
    t_eval=t_eval,
)

e0 = energy(sol.y[:, 0])
print("state layout: [q..., v...] = [x, y, z, vx, vy, vz]")
print(f"status: {sol.message}")
for i, t in enumerate(sol.t):
    q = sol.y[:3, i]
    v = sol.y[3:, i]
    print(
        f"t = {t:>5.2f}, q = {q}, v = {v}, dE = {energy(np.concatenate([q, v])) - e0:>9.2e}"
    )
