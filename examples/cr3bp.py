"""
Arenstorf orbit in the Circular Restricted Three-Body Problem (CR3BP).
This is a famous periodic orbit discovered by Richard Arenstorf, used as a
benchmark problem in numerical ODE literature (Hairer, Norsett & Wanner).
"""
import ivp
import numpy as np
import matplotlib.pyplot as plt


def cr3bp(t, state, mu):
    """CR3BP equations of motion in the rotating frame."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    ax = x + 2*vy - (1-mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = y - 2*vx - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
    
    return [vx, vy, vz, ax, ay, az]


def jacobi_constant(state, mu):
    """Calculate Jacobi constant (should be conserved)."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    U = 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2
    return 2*U - (vx**2 + vy**2 + vz**2)


# Earth-Moon mass ratio
mu = 0.012277471

# Arenstorf orbit initial conditions (periodic orbit, period T ~ 17.0652)
# From Hairer, Norsett & Wanner "Solving ODEs I"
x0 = 0.994
vy0 = -2.00158510637908252240537862224
state0 = [x0, 0, 0, 0, vy0, 0]
period = 17.0652165601579625588917206249

print("Arenstorf Orbit (Earth-Moon CR3BP)")
print("=" * 40)

sol = ivp.solve_ivp(
    cr3bp, (0, period), state0,
    method='DOP853', args=(mu,),
    rtol=1e-12, atol=1e-14,
    dense_output=True
)

C_initial = jacobi_constant(state0, mu)
C_final = jacobi_constant([sol.y[i][-1] for i in range(6)], mu)
final_state = [sol.y[i][-1] for i in range(6)]

print(f"Status: {sol.message}")
print(f"nfev: {sol.nfev}, steps: {len(sol.t)}")
print(f"Jacobi constant error: {abs(C_final - C_initial):.2e}")
print(f"Position error at T: dx={abs(final_state[0] - state0[0]):.2e}, dy={abs(final_state[1] - state0[1]):.2e}")

# Plot the periodic orbit
fig, ax = plt.subplots(figsize=(10, 8))

# Plot trajectory
t_plot = np.linspace(0, period, 1000)
traj = sol.sol(t_plot)
ax.plot(traj[0], traj[1], 'b-', linewidth=1.5, label='Arenstorf orbit')

# Mark Earth and Moon
ax.plot(-mu, 0, 'go', markersize=15, label='Earth')
ax.plot(1-mu, 0, 'ko', markersize=8, label='Moon')

# Mark start position
ax.plot(state0[0], state0[1], 'r*', markersize=12, label='Start')

ax.set_xlabel('x (normalized)')
ax.set_ylabel('y (normalized)')
ax.set_title('Arenstorf Periodic Orbit (Earth-Moon System)')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.show()
