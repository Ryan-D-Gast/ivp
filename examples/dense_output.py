"""Dense output demonstration with Lorenz system."""
import ivp
import numpy as np
import matplotlib.pyplot as plt

def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

sigma, rho, beta = 10.0, 28.0, 8.0/3.0
y0 = [1.0, 1.0, 1.0]

sol = ivp.solve_ivp(lorenz, (0, 50), y0, method='DOP853',
                    args=(sigma, rho, beta), rtol=1e-10, atol=1e-12,
                    dense_output=True)

print(f"Solver steps: {len(sol.t)}")
print(f"Dense output available: {sol.sol is not None}")

# Sample 10000 points using dense output
t_fine = np.linspace(0, 50, 10000)
y_fine = sol.sol(t_fine)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(y_fine[0], y_fine[1], y_fine[2], 'b-', linewidth=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Lorenz Attractor')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(y_fine[0], y_fine[2], 'b-', linewidth=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_title('x-z projection')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
