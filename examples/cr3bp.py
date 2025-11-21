"""
Example of solving the Circular Restricted Three-Body Problem (CR3BP).
Demonstrates:
- Solving a system of 6 equations
- Using high-order explicit methods (DOP853)
- Event detection (crossing the y-axis)
- High precision tolerances
"""
import ivp
import numpy as np
import matplotlib.pyplot as plt

def cr3bp(t, sv, mu):
    x, y, z, vx, vy, vz = sv
    
    r13 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r23 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)
    
    dx = vx
    dy = vy
    dz = vz
    dvx = x + 2.0 * vy - (1.0 - mu) * (x + mu) / r13**3 - mu * (x - 1.0 + mu) / r23**3
    dvy = y - 2.0 * vx - (1.0 - mu) * y / r13**3 - mu * y / r23**3
    dvz = -(1.0 - mu) * vz / r13**3 - mu * vz / r23**3
    
    return [dx, dy, dz, dvx, dvy, dvz]

def event_y_cross(t, sv, mu):
    return sv[1]

event_y_cross.terminal = True
event_y_cross.direction = 1

mu = 0.1
t_span = (0, 10.0)
y0 = [0.5, 0.1, 0.0, 0.0, 1.2, 0.0]
t_eval = np.linspace(0, 10, 11)

sol = ivp.solve_ivp(cr3bp, t_span, y0, method='DOP853', t_eval=t_eval, args=(mu,), events=event_y_cross, rtol=1e-6, atol=1e-9)

print("Status:", sol.message)
print("nfev:", sol.nfev)

if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
    print("Event detected at t =", sol.t_events[0][0])
    print("State at event:", sol.y_events[0][0])

# Plotting
plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('CR3BP Trajectory')
plt.grid(True)
plt.axis('equal')
plt.show()
