"""
Comprehensive demonstration of advanced features.
Demonstrates:
- Dense output (continuous solution)
- Multiple event tracking (ground impact and apex)
- Event direction and terminal attributes
- Accessing event results
"""
import ivp
import numpy as np
import matplotlib.pyplot as plt

def cannon_with_drag(t, y, k):
    """
    Equations of motion for a cannonball with air drag.
    y[0] = x
    y[1] = y (height)
    y[2] = vx
    y[3] = vy
    
    Drag force is proportional to velocity squared: Fd = -k * v * |v|
    """
    x, h, vx, vy = y
    v_sq = vx**2 + vy**2
    v = np.sqrt(v_sq)
    
    ax = -k * vx * v
    ay = -9.81 - k * vy * v
    
    return [vx, vy, ax, ay]

def hit_ground(t, y, k):
    """Event: Cannonball hits the ground (y=0)."""
    return y[1]

hit_ground.terminal = True
hit_ground.direction = -1

def apex(t, y, k):
    """Event: Cannonball reaches highest point (vy=0)."""
    return y[3]

apex.direction = -1 # Going from positive vertical velocity to negative

# Parameters
k = 0.05 # Drag coefficient
y0 = [0, 0, 50, 50] # Initial state: x=0, y=0, vx=50, vy=50
t_span = (0, 20)

# Solve with dense output and events
sol = ivp.solve_ivp(
    cannon_with_drag, 
    t_span, 
    y0, 
    method='DOP853', 
    args=(k,), 
    events=[hit_ground, apex], 
    dense_output=True,
    rtol=1e-8,
    atol=1e-8
)

print(f"Status: {sol.message}")
print(f"Number of steps: {len(sol.t)}")
print(f"Function evaluations: {sol.nfev}")

# Analyze events
t_events = sol.t_events
y_events = sol.y_events

if len(t_events[1]) > 0:
    t_apex = t_events[1][0]
    y_apex = y_events[1][0]
    print(f"Apex reached at t={t_apex:.4f}s, height={y_apex[1]:.4f}m")

if len(t_events[0]) > 0:
    t_impact = t_events[0][0]
    y_impact = y_events[0][0]
    print(f"Impact at t={t_impact:.4f}s, distance={y_impact[0]:.4f}m")

# Use dense output to sample smoothly for plotting
t_plot = np.linspace(0, t_impact, 200)
y_plot = sol.sol(t_plot) # Interpolated solution

plt.figure(figsize=(10, 6))
plt.plot(y_plot[0], y_plot[1], label='Trajectory')
plt.plot(y_apex[0], y_apex[1], 'ro', label='Apex')
plt.plot(y_impact[0], y_impact[1], 'ko', label='Impact')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Cannonball with Air Drag (Dense Output & Events)')
plt.legend()
plt.grid(True)
plt.show()
