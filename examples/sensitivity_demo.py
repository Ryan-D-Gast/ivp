import numpy as np
from ivp import solve_ivp

def system(t, y, p):
    """
    Simple linear system: y' = p[0] * y
    Analytic solution: y(t) = y0 * exp(p[0] * t)
    Sensitivity dy/dp[0] = t * y(t)
    """
    return p[0] * y

def quadrature_fun(t, y, p):
    """
    Integrate y over time: G = integral(y dt)
    Analytic: (y0/p0) * (exp(p0*t) - 1)
    """
    return y

# Parameters and initial conditions (Testing LIST input)
p = [0.5]
y0 = [1.0]
t_span = (0.0, 2.0)

print("--- 1. Forward Sensitivity Analysis & Quadrature ---")
# Solve with forward sensitivities and quadrature
sol = solve_ivp(
    system, 
    t_span, 
    y0, 
    p=p, 
    quadrature=quadrature_fun, 
    forward_sensitivity=True,
    dense_output=True
)

# 1a. Check Quadrature
# Integral of exp(0.5*t) from 0 to 2 is (exp(1)-1)/0.5 = 2*(e-1) approx 3.4365
analytic_quad = (y0[0] / p[0]) * (np.exp(p[0] * t_span[1]) - 1.0)
print(f"Numerical Quadrature: {sol.quad[0]:.6f}")
print(f"Analytic Quadrature:  {analytic_quad:.6f}")

# 1b. Check Forward Sensitivities
# dy/dp at t=2 is 2 * exp(1) approx 5.4365
analytic_sens = t_span[1] * y0[0] * np.exp(p[0] * t_span[1])
numerical_sens = sol.s[0, 0, -1]
print(f"Numerical Sensitivity dy/dp[0] at t=2: {numerical_sens:.6f}")
print(f"Analytic Sensitivity dy/dp[0] at t=2:  {analytic_sens:.6f}")

print("\n--- 2. Adjoint Sensitivity Analysis ---")
# Compute gradient of Cost G = integral(y dt)
# dG/dp[0] should match the integral of dy/dp[0] over time
# Integral(t * exp(0.5*t) dt) from 0 to 2
# = [2*t*exp(0.5*t) - 4*exp(0.5*t)] from 0 to 2
# = (4*e - 4*e) - (0 - 4) = 4

# For Adjoint solve:
# Testing LIST input for lambda_tf and LIST returns from callbacks
lambda_tf = [0.0]
dgdy = lambda t, y, p: [1.0]
dgdp = lambda t, y, p: [0.0]
dhdp = lambda y_tf, p: [0.0]

grad = sol.adjoint_solve(lambda_tf, dgdy, dgdp, dhdp)
print(f"Adjoint Gradient dG/dp[0]: {grad[0]:.6f}")
print(f"Analytic Gradient:         4.000000")

print("\n--- 3. More Complex Cost Function (Adjoint) ---")
# Cost G = 0.5 * y(tf)^2
# lambda_tf = dG/dy(tf) = y(tf)
# dg/dy = 0, dg/dp = 0, dh/dp = 0

y_tf = sol.y[0, -1]
lambda_tf_2 = [y_tf]
dgdy_2 = lambda t, y, p: [0.0]
dgdp_2 = lambda t, y, p: [0.0]
dhdp_2 = lambda y_tf, p: [0.0]

grad_2 = sol.adjoint_solve(lambda_tf_2, dgdy_2, dgdp_2, dhdp_2)
# Analytic: d(0.5 * (y0*exp(p0*t))^2)/dp0 = y0^2 * t * exp(2*p0*t)
# at t=2, p0=0.5: 1^2 * 2 * exp(2) = 2*e^2 approx 14.778
analytic_grad_2 = y0[0]**2 * t_span[1] * np.exp(2.0 * p[0] * t_span[1])
print(f"Adjoint Gradient d(0.5*y_tf^2)/dp[0]: {grad_2[0]:.6f}")
print(f"Analytic Gradient:                   {analytic_grad_2:.6f}")
