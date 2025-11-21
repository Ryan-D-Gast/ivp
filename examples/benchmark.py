import time
import numpy as np
import ivp
from scipy.integrate import solve_ivp as scipy_solve_ivp

def van_der_pol(t, y, eps):
    y0, y1 = y
    dy0 = y1
    dy1 = ((1.0 - y0**2) * y1 - y0) / eps
    return [dy0, dy1]

def linear_system(t, y):
    return -y

def benchmark_problem(name, fun, t_span, y0, args, method_ivp, method_scipy, rtol=1e-6, atol=1e-6):
    print(f"\nBenchmarking {name}...")
    
    # Warmup
    try:
        ivp.solve_ivp(fun, t_span, y0, method=method_ivp, args=args, rtol=rtol, atol=atol)
    except Exception as e:
        print(f"  IVP Warmup failed: {e}")
        return

    try:
        scipy_solve_ivp(fun, t_span, y0, method=method_scipy, args=args, rtol=rtol, atol=atol)
    except Exception as e:
        print(f"  SciPy Warmup failed: {e}")
        return

    # IVP (Rust)
    start = time.perf_counter()
    sol_ivp = ivp.solve_ivp(fun, t_span, y0, method=method_ivp, args=args, rtol=rtol, atol=atol)
    end = time.perf_counter()
    time_ivp = end - start
    print(f"  IVP ({method_ivp}): {time_ivp:.6f} s, nfev={sol_ivp.nfev}, success={sol_ivp.success}")

    # SciPy
    start = time.perf_counter()
    sol_scipy = scipy_solve_ivp(fun, t_span, y0, method=method_scipy, args=args, rtol=rtol, atol=atol)
    end = time.perf_counter()
    time_scipy = end - start
    print(f"  SciPy ({method_scipy}): {time_scipy:.6f} s, nfev={sol_scipy.nfev}, success={sol_scipy.success}")

    if time_ivp > 0:
        print(f"  Speedup: {time_scipy / time_ivp:.2f}x")

if __name__ == "__main__":
    print("Comparing ivp (Rust) vs scipy.integrate.solve_ivp")
    
    # Problem 1: Non-stiff Van der Pol (DOP853)
    # High precision integration over a long interval
    eps = 1.0
    t_span = (0, 100.0)
    y0 = [2.0, 0.0]
    benchmark_problem("Van der Pol (Non-stiff, eps=1.0, DOP853)", van_der_pol, t_span, y0, (eps,), 'DOP853', 'DOP853')

    # Problem 2: Non-stiff Van der Pol (RK45)
    benchmark_problem("Van der Pol (Non-stiff, eps=1.0, RK45)", van_der_pol, t_span, y0, (eps,), 'RK45', 'RK45')

    # Problem 3: Stiff Van der Pol (BDF)
    eps = 1e-3
    t_span = (0, 2.0)
    y0 = [2.0, 0.0]
    benchmark_problem("Van der Pol (Stiff, eps=1e-3, BDF)", van_der_pol, t_span, y0, (eps,), 'BDF', 'BDF', rtol=1e-6, atol=1e-6)

    # Problem 4: Large Linear System (RK45)
    # Tests overhead of passing large arrays back and forth
    N = 1000
    y0_lin = np.random.rand(N)
    t_span_lin = (0, 10.0)
    benchmark_problem(f"Linear System (N={N}, RK45)", linear_system, t_span_lin, y0_lin, (), 'RK45', 'RK45')
