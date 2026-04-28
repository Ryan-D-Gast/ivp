import numpy as np
from ivp import solve_ivp
import matplotlib.pyplot as plt

def damped_oscillator(t, y, p):
    """
    Damped linear oscillator:
    y'' + 2*zeta*omega*y' + omega^2*y = 0
    State: [q, v]
    Parameters: [zeta, omega]
    """
    q, v = y
    zeta, omega = p
    dqdt = v
    dvdt = -2.0 * zeta * omega * v - omega**2 * q
    return [dqdt, dvdt]

def energy_quad(t, y, p):
    """
    Quadrature: Total energy E = 0.5 * v^2 + 0.5 * omega^2 * q^2
    Let's integrate the squared displacement (useful for cost functions).
    """
    q, v = y
    return [q**2]

# Parameters: zeta (damping), omega (frequency)
p = np.array([0.1, 2.0])
y0 = [1.0, 0.0]
t_span = (0.0, 10.0)

# Define a dense grid for smooth plotting
t_plot = np.linspace(t_span[0], t_span[1], 500)

print(f"Solving damped oscillator with p={p}...")
sol = solve_ivp(
    damped_oscillator,
    t_span,
    y0,
    p=p,
    quadrature=energy_quad,
    forward_sensitivity=True,
    dense_output=True,
    t_eval=t_plot  # Use t_eval to get dense, smooth data for plotting
)

print(f"Success: {sol.success}")
print(f"Final state: {sol.y[:, -1]}")
print(f"Integral of q^2: {sol.quad[0]:.6f}")

# Plot results
t = sol.t
q = sol.y[0, :]
v = sol.y[1, :]

plt.figure(figsize=(12, 8))

# Subplot 1: Trajectory
plt.subplot(2, 2, 1)
plt.plot(t, q, label='q (displacement)')
plt.plot(t, v, label='v (velocity)')
plt.title("Damped Oscillator Trajectory (Smooth)")
plt.xlabel("Time")
plt.legend()

# Subplot 2: Sensitivities of q w.r.t zeta and omega
plt.subplot(2, 2, 2)
# sol.s has shape (n_params, n_states, n_points)
dq_dzeta = sol.s[0, 0, :]
dq_domega = sol.s[1, 0, :]
plt.plot(t, dq_dzeta, label='dq/dzeta')
plt.plot(t, dq_domega, label='dq/domega')
plt.title("Forward Sensitivities (Smooth)")
plt.xlabel("Time")
plt.legend()

# Subplot 3: Adjoint Gradient check
lambda_tf = np.array([0.0, 0.0]) # No terminal cost
def dgdy(t, y, p):
    q, v = y
    return [2.0 * q, 0.0] # d(q^2)/dy = [2q, 0]

def dgdp(t, y, p):
    return [0.0, 0.0]

def dhdp(y_tf, p):
    return [0.0, 0.0]

grad_adj = sol.adjoint_solve(lambda_tf, dgdy, dgdp, dhdp)
print(f"Adjoint Gradient dG/dp: {grad_adj}")

# Verify with forward sensitivities
grad_fwd = []
for p_idx in range(len(p)):
    g_p = np.trapezoid(2.0 * q * sol.s[p_idx, 0, :], t)
    grad_fwd.append(g_p)

print(f"Forward Sensitivity Gradient (trapezoid): {grad_fwd}")

plt.subplot(2, 2, 3)
plt.bar(['dG/dzeta', 'dG/domega'], grad_adj, alpha=0.5, label='Adjoint')
plt.bar(['dG/dzeta', 'dG/domega'], grad_fwd, alpha=0.5, label='Forward (integral)')
plt.title("Cost Gradient Comparison")
plt.legend()

plt.tight_layout()
output_path = "examples/oscillator_sensitivity.png"
plt.savefig(output_path)
print(f"Saved {output_path}")
