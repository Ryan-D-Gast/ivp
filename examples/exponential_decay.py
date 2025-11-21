"""
Basic example of solving a scalar ODE.
Demonstrates:
- Basic usage of solve_ivp
- Simple system definition
- Plotting results
"""
import ivp
import numpy as np
import matplotlib.pyplot as plt

def exponential_decay(t, y):
    return -0.5 * y

# Basic exponential decay
t_span = (0, 10)
y0 = [2, 4, 8]
sol = ivp.solve_ivp(exponential_decay, t_span, y0)

print("Status:", sol.message)
print("Time points:", sol.t)
print("Values:", sol.y)

# Plotting
plt.figure()
plt.plot(sol.t, sol.y.T)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Exponential Decay')
plt.legend(['y1', 'y2', 'y3'])
plt.grid(True)
plt.show()
