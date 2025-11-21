"""
Example of solving a stiff ODE system (Van der Pol oscillator).
Demonstrates:
- Using implicit methods (BDF) for stiff problems
- Passing parameters to the ODE function (args)
- Specifying t_eval for custom output points
"""
import ivp
import numpy as np
import matplotlib.pyplot as plt

def van_der_pol(t, y, eps):
    y0, y1 = y
    dy0 = y1
    dy1 = ((1.0 - y0**2) * y1 - y0) / eps
    return [dy0, dy1]

eps = 1e-3
t_span = (0, 2.0)
y0 = [2.0, 0.0]
t_eval = np.linspace(0, 2.0, 21)

# Using BDF for stiff problem
sol = ivp.solve_ivp(van_der_pol, t_span, y0, method='BDF', t_eval=t_eval, args=(eps,), rtol=1e-9, atol=1e-9)

print("Status:", sol.message)
print("nfev:", sol.nfev)
print("njev:", sol.njev)
print("nlu:", sol.nlu)

# Plotting
plt.figure()
plt.plot(sol.t, sol.y[0], label='y0')
plt.plot(sol.t, sol.y[1], label='y1')
plt.xlabel('t')
plt.legend()
plt.title(f'Van der Pol (eps={eps})')
plt.grid(True)
plt.show()
