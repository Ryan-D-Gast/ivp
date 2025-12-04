"""Exponential Decay - Basic ODE Example

Solves dy/dt = -k*y with analytical comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import ivp

k = 0.5
y0 = [10.0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 21)

sol = ivp.solve_ivp(
    lambda t, y: [-k * y[0]],
    t_span, y0,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-8, atol=1e-10
)

y_analytical = y0[0] * np.exp(-k * sol.t)
max_error = np.max(np.abs(sol.y[0] - y_analytical))

print(f"Exponential Decay: dy/dt = -{k}*y, y(0) = {y0[0]}")
print(f"Status: {sol.message}")
print(f"Max error vs analytical: {max_error:.2e}")

plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], 'b.-', label='Numerical')
plt.plot(sol.t, y_analytical, 'r--', label='Analytical')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Exponential Decay')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
