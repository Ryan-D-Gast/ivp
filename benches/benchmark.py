"""
Benchmark: ivp (Rust) vs SciPy
==============================

This script compares the performance of the ivp library (Rust with PyO3 bindings)
against SciPy's solve_ivp for various ODE problems.

Note: Performance depends on many factors including:
- Problem size and complexity
- Requested tolerances
- Number of function evaluations needed
- Python/NumPy overhead for function calls

The Rust library may be faster for some problems but has Python call overhead.
"""
import time
import numpy as np
import ivp
from scipy.integrate import solve_ivp as scipy_solve_ivp


def van_der_pol(t, y, mu):
    """Van der Pol oscillator - standard test problem."""
    y0, y1 = y
    dy0 = y1
    dy1 = mu * (1.0 - y0**2) * y1 - y0
    return [dy0, dy1]


def lorenz(t, y, sigma, rho, beta):
    """Lorenz system - chaotic attractor."""
    x, y_val, z = y
    return [
        sigma * (y_val - x),
        x * (rho - z) - y_val,
        x * y_val - beta * z
    ]


def linear_system(t, y):
    """Simple linear decay: dy/dt = -y"""
    return -y


def run_benchmark(name, fun, t_span, y0, args, methods, rtol=1e-6, atol=1e-8, n_runs=5):
    """Run benchmark comparing ivp and scipy for given methods."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"t_span: {t_span}, dim: {len(y0)}, rtol: {rtol}, atol: {atol}")
    print(f"Averaging over {n_runs} runs\n")
    
    results = {}
    
    for method in methods:
        # Warmup
        try:
            ivp.solve_ivp(fun, t_span, y0, method=method, args=args, rtol=rtol, atol=atol)
            scipy_solve_ivp(fun, t_span, y0, method=method, args=args, rtol=rtol, atol=atol)
        except Exception as e:
            print(f"  {method}: Warmup failed - {e}")
            continue
        
        # Benchmark ivp (Rust)
        times_ivp = []
        for _ in range(n_runs):
            start = time.perf_counter()
            sol_ivp = ivp.solve_ivp(fun, t_span, y0, method=method, args=args, rtol=rtol, atol=atol)
            times_ivp.append(time.perf_counter() - start)
        time_ivp = np.median(times_ivp)
        
        # Benchmark scipy
        times_scipy = []
        for _ in range(n_runs):
            start = time.perf_counter()
            sol_scipy = scipy_solve_ivp(fun, t_span, y0, method=method, args=args, rtol=rtol, atol=atol)
            times_scipy.append(time.perf_counter() - start)
        time_scipy = np.median(times_scipy)
        
        speedup = time_scipy / time_ivp if time_ivp > 0 else 0
        
        results[method] = {
            'ivp_time': time_ivp,
            'scipy_time': time_scipy,
            'speedup': speedup,
            'ivp_nfev': sol_ivp.nfev,
            'scipy_nfev': sol_scipy.nfev
        }
        
        print(f"{method}:")
        print(f"  ivp:   {time_ivp*1000:8.2f} ms  (nfev={sol_ivp.nfev})")
        print(f"  scipy: {time_scipy*1000:8.2f} ms  (nfev={sol_scipy.nfev})")
        if speedup >= 1:
            print(f"  ivp is {speedup:.2f}x faster")
        else:
            print(f"  scipy is {1/speedup:.2f}x faster")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("IVP (Rust) vs SciPy Benchmark")
    print("=" * 60)
    
    # Problem 1: Van der Pol (non-stiff)
    run_benchmark(
        "Van der Pol (non-stiff, μ=1)",
        van_der_pol,
        t_span=(0, 100.0),
        y0=[2.0, 0.0],
        args=(1.0,),
        methods=['RK45', 'DOP853'],
        rtol=1e-6, atol=1e-8
    )
    
    # Problem 2: Van der Pol (stiff)
    run_benchmark(
        "Van der Pol (stiff, μ=1000)",
        van_der_pol,
        t_span=(0, 3000.0),
        y0=[2.0, 0.0],
        args=(1000.0,),
        methods=['BDF', 'Radau'],
        rtol=1e-4, atol=1e-6
    )
    
    # Problem 3: Lorenz system
    run_benchmark(
        "Lorenz System (chaotic)",
        lorenz,
        t_span=(0, 100.0),
        y0=[1.0, 1.0, 1.0],
        args=(10.0, 28.0, 8.0/3.0),
        methods=['RK45', 'DOP853'],
        rtol=1e-8, atol=1e-10
    )
    
    # Problem 4: Large linear system (test overhead)
    run_benchmark(
        "Large Linear System (N=100)",
        linear_system,
        t_span=(0, 10.0),
        y0=np.ones(100),
        args=(),
        methods=['RK45'],
        rtol=1e-6, atol=1e-8
    )

    print("\n" + "=" * 60)
    print("NOTES")
    print("=" * 60)
    print("""
- Performance varies with problem type and size
- Rust library has Python call overhead for the ODE function
- For pure Python ODE functions, overhead may negate Rust speed
- Best speedups occur with many internal solver steps
- Try with your specific problem to evaluate performance
""")
