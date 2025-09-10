<p align="center">
  <img src="./assets/logo.svg" width="1000" alt="ivp">
</p>

<p align="center">
    <a href="https://crates.io/crates/ivp">
        <img src="https://img.shields.io/crates/v/ivp.svg?style=flat-square" alt="crates.io">
    </a>
    <a href="https://docs.rs/ivp">
        <img src="https://docs.rs/ivp/badge.svg" alt="docs.rs">
    </a>
    <a href="https://github.com/Ryan-D-Gast/ivp/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
</p>

<p align="center">
    <strong>
        <a href="https://docs.rs/ivp/latest/ivp/">Documentation</a> |
        <a href="./examples/">Examples</a> |
        <a href="https://github.com/Ryan-D-Gast/ivp">GitHub</a> |
        <a href="https://crates.io/crates/ivp">Crates.io</a>
    </strong>
</p>

-----

<p align="center">
<b>A library of numerical methods for solving initial value problems (IVPs)</b><br>
<i>for the Rust programming language.</i>
</p>

-----

This library provides a pure rust implementation of SciPy's `solve_ivp` function with slight modifications to the API to better fit Rust's design patterns.

Currently implemented solvers:
-   **DOP853**: An 8th order Dormand-Prince method with step-size control and dense output.
-   **DOPRI5**: A 5th order Dormand-Prince method with step-size control and dense output.
-   **RK4**: The classic 4th order Runge-Kutta method with fixed step-size and cubic Hermite interpolation for dense output.
-   **RK23**: A 3rd order Runge-Kutta method with 2nd order error estimate for step-size control.

Planned additions in development:
-   **Radau**: A 5th order implicit Runge-Kutta method of Radau IIA type with step-size control and dense output.
-   **BDF**: A variable-order (1 to 5) Backward Differentiation Formula method with step-size control and dense output.
