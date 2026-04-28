//! Typed IVP builders for the public Rust API.

use crate::{
    Float,
    error::Error,
    ivp::{FirstOrderSystem, SecondOrderSystem, SeparableHamiltonianSystem},
    matrix::MatrixStorage,
    methods::{SymplecticMethod, Tolerance},
};

use super::{
    SymplecticConfig,
    options::{FirstOrderConfig, JacobianSource, Method},
    solution::Solution,
    solve_first_order_impl, solve_hamiltonian_impl, solve_second_order_impl,
};

/// Entry point for constructing typed initial value problems.
pub struct Ivp;

impl Ivp {
    /// Create a builder for a first-order system `y' = f(t, y, p)`.
    pub fn first_order<'a, F>(
        system: &'a F,
        t0: Float,
        tf: Float,
        y0: &'a [Float],
    ) -> FirstOrderIvp<'a, F>
    where
        F: FirstOrderSystem,
    {
        FirstOrderIvp {
            system,
            t0,
            tf,
            y0,
            p: Vec::new(),
            method: Method::DOPRI5,
            rtol: Tolerance::Scalar(1e-3),
            atol: Tolerance::Scalar(1e-6),
            max_steps: None,
            t_eval: None,
            first_step: None,
            max_step: None,
            min_step: None,
            dense_output: false,
            jac_storage: MatrixStorage::Full,
            jacobian_source: JacobianSource::Auto,
            mass_storage: MatrixStorage::Identity,
            nind1: None,
            nind2: None,
            nind3: None,
        }
    }

    /// Create a builder for a second-order system `q'' = a(t, q, p)`.
    pub fn second_order<'a, F>(
        system: &'a F,
        t0: Float,
        tf: Float,
        q0: &'a [Float],
        v0: &'a [Float],
    ) -> SecondOrderIvp<'a, F>
    where
        F: SecondOrderSystem,
    {
        SecondOrderIvp {
            system,
            t0,
            tf,
            q0,
            v0,
            p: Vec::new(),
            method: SymplecticMethod::VelocityVerlet,
            step_size: None,
            max_steps: None,
            t_eval: None,
            dense_output: false,
        }
    }

    /// Create a builder for a separable Hamiltonian system in canonical form.
    pub fn hamiltonian<'a, F>(
        system: &'a F,
        t0: Float,
        tf: Float,
        q0: &'a [Float],
        p0: &'a [Float],
    ) -> HamiltonianIvp<'a, F>
    where
        F: SeparableHamiltonianSystem,
    {
        HamiltonianIvp {
            system,
            t0,
            tf,
            q0,
            p0,
            p: Vec::new(),
            method: SymplecticMethod::VelocityVerlet,
            step_size: None,
            max_steps: None,
            t_eval: None,
            dense_output: false,
        }
    }
}

/// Builder for first-order initial value problems.
#[derive(Clone, Debug)]
pub struct FirstOrderIvp<'a, F> {
    system: &'a F,
    t0: Float,
    tf: Float,
    y0: &'a [Float],
    p: Vec<Float>,
    method: Method,
    rtol: Tolerance,
    atol: Tolerance,
    max_steps: Option<usize>,
    t_eval: Option<Vec<Float>>,
    first_step: Option<Float>,
    max_step: Option<Float>,
    min_step: Option<Float>,
    dense_output: bool,
    jac_storage: MatrixStorage,
    jacobian_source: JacobianSource,
    mass_storage: MatrixStorage,
    nind1: Option<usize>,
    nind2: Option<usize>,
    nind3: Option<usize>,
}

impl<'a, F> FirstOrderIvp<'a, F> {
    /// Set the numerical method used to solve the IVP.
    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Set the relative tolerance used for adaptive error control.
    pub fn rtol<T>(mut self, rtol: T) -> Self
    where
        T: Into<Tolerance>,
    {
        self.rtol = rtol.into();
        self
    }

    /// Set the absolute tolerance used for adaptive error control.
    pub fn atol<T>(mut self, atol: T) -> Self
    where
        T: Into<Tolerance>,
    {
        self.atol = atol.into();
        self
    }

    /// Limit the maximum number of accepted or attempted solver steps.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_max_steps(mut self, max_steps: Option<usize>) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Request output at specific times instead of at all accepted internal steps.
    pub fn t_eval(mut self, t_eval: Vec<Float>) -> Self {
        self.t_eval = Some(t_eval);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_t_eval(mut self, t_eval: Option<Vec<Float>>) -> Self {
        self.t_eval = t_eval;
        self
    }

    /// Set the initial step size.
    pub fn first_step(mut self, first_step: Float) -> Self {
        self.first_step = Some(first_step);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_first_step(mut self, first_step: Option<Float>) -> Self {
        self.first_step = first_step;
        self
    }

    /// Set an upper bound on the step size.
    pub fn max_step(mut self, max_step: Float) -> Self {
        self.max_step = Some(max_step);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_max_step(mut self, max_step: Option<Float>) -> Self {
        self.max_step = max_step;
        self
    }

    /// Set a lower bound on the step size.
    pub fn min_step(mut self, min_step: Float) -> Self {
        self.min_step = Some(min_step);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_min_step(mut self, min_step: Option<Float>) -> Self {
        self.min_step = min_step;
        self
    }

    /// Enable or disable dense output interpolation.
    pub fn dense_output(mut self, dense_output: bool) -> Self {
        self.dense_output = dense_output;
        self
    }

    /// Select the storage layout used for the Jacobian matrix.
    pub fn jac_storage(mut self, jac_storage: MatrixStorage) -> Self {
        self.jac_storage = jac_storage;
        self
    }

    /// Select how the solver should obtain Jacobian information.
    ///
    /// This mainly matters for LSODA, which can either use its internal
    /// finite-difference Jacobian logic or call the system's `jac(...)`
    /// implementation directly.
    pub fn jacobian_source(mut self, jacobian_source: JacobianSource) -> Self {
        self.jacobian_source = jacobian_source;
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_jacobian_source(mut self, jacobian_source: Option<JacobianSource>) -> Self {
        if let Some(jacobian_source) = jacobian_source {
            self.jacobian_source = jacobian_source;
        }
        self
    }

    /// Select the storage layout used for the mass matrix.
    pub fn mass_storage(mut self, mass_storage: MatrixStorage) -> Self {
        self.mass_storage = mass_storage;
        self
    }

    /// Set the number of index-1 variables in the DAE partition.
    pub fn nind1(mut self, nind1: usize) -> Self {
        self.nind1 = Some(nind1);
        self
    }

    /// Set the number of index-2 variables in the DAE partition.
    pub fn nind2(mut self, nind2: usize) -> Self {
        self.nind2 = Some(nind2);
        self
    }

    /// Set the number of index-3 variables in the DAE partition.
    pub fn nind3(mut self, nind3: usize) -> Self {
        self.nind3 = Some(nind3);
        self
    }

    /// Set the parameters passed to the system functions.
    pub fn p(mut self, p: Vec<Float>) -> Self {
        self.p = p;
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_p(mut self, p: Option<Vec<Float>>) -> Self {
        if let Some(p) = p {
            self.p = p;
        }
        self
    }
}

impl<F> FirstOrderIvp<'_, F>
where
    F: FirstOrderSystem,
{
    /// Solve the configured first-order IVP.
    pub fn solve(self) -> Result<Solution, Error> {
        let config = FirstOrderConfig {
            p: self.p,
            method: self.method,
            rtol: self.rtol,
            atol: self.atol,
            max_steps: self.max_steps,
            t_eval: self.t_eval,
            first_step: self.first_step,
            max_step: self.max_step,
            min_step: self.min_step,
            dense_output: self.dense_output,
            jac_storage: self.jac_storage,
            jacobian_source: self.jacobian_source,
            mass_storage: self.mass_storage,
            nind1: self.nind1,
            nind2: self.nind2,
            nind3: self.nind3,
        };
        solve_first_order_impl(self.system, self.t0, self.tf, self.y0, config)
    }
}

/// Builder for second-order symplectic IVPs.
#[derive(Clone, Debug)]
pub struct SecondOrderIvp<'a, F> {
    system: &'a F,
    t0: Float,
    tf: Float,
    q0: &'a [Float],
    v0: &'a [Float],
    p: Vec<Float>,
    method: SymplecticMethod,
    step_size: Option<Float>,
    max_steps: Option<usize>,
    t_eval: Option<Vec<Float>>,
    dense_output: bool,
}

impl<'a, F> SecondOrderIvp<'a, F> {
    /// Set the symplectic method used to solve the IVP.
    pub fn method(mut self, method: SymplecticMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the fixed symplectic step size.
    pub fn step_size(mut self, step_size: Float) -> Self {
        self.step_size = Some(step_size);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_step_size(mut self, step_size: Option<Float>) -> Self {
        self.step_size = step_size;
        self
    }

    /// Limit the maximum number of symplectic substeps.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_max_steps(mut self, max_steps: Option<usize>) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Request output at specific times instead of at every internal step endpoint.
    pub fn t_eval(mut self, t_eval: Vec<Float>) -> Self {
        self.t_eval = Some(t_eval);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_t_eval(mut self, t_eval: Option<Vec<Float>>) -> Self {
        self.t_eval = t_eval;
        self
    }

    /// Enable or disable dense output interpolation.
    pub fn dense_output(mut self, dense_output: bool) -> Self {
        self.dense_output = dense_output;
        self
    }

    /// Set the parameters passed to the system functions.
    pub fn p(mut self, p: Vec<Float>) -> Self {
        self.p = p;
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_p(mut self, p: Option<Vec<Float>>) -> Self {
        if let Some(p) = p {
            self.p = p;
        }
        self
    }
}

impl<F> SecondOrderIvp<'_, F>
where
    F: SecondOrderSystem,
{
    /// Solve the configured second-order IVP.
    pub fn solve(self) -> Result<Solution, Error> {
        let config = SymplecticConfig {
            method: self.method,
            step_size: self
                .step_size
                .unwrap_or_else(|| (self.tf - self.t0) / 100.0),
            max_steps: self.max_steps,
            t_eval: self.t_eval,
            dense_output: self.dense_output,
            p: self.p,
        };
        solve_second_order_impl(self.system, self.t0, self.tf, self.q0, self.v0, config)
    }
}

/// Builder for separable Hamiltonian IVPs.
#[derive(Clone, Debug)]
pub struct HamiltonianIvp<'a, F> {
    system: &'a F,
    t0: Float,
    tf: Float,
    q0: &'a [Float],
    p0: &'a [Float],
    p: Vec<Float>,
    method: SymplecticMethod,
    step_size: Option<Float>,
    max_steps: Option<usize>,
    t_eval: Option<Vec<Float>>,
    dense_output: bool,
}

impl<'a, F> HamiltonianIvp<'a, F> {
    /// Set the symplectic method used to solve the IVP.
    pub fn method(mut self, method: SymplecticMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the fixed symplectic step size.
    pub fn step_size(mut self, step_size: Float) -> Self {
        self.step_size = Some(step_size);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_step_size(mut self, step_size: Option<Float>) -> Self {
        self.step_size = step_size;
        self
    }

    /// Limit the maximum number of symplectic substeps.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_max_steps(mut self, max_steps: Option<usize>) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Request output at specific times instead of at every internal step endpoint.
    pub fn t_eval(mut self, t_eval: Vec<Float>) -> Self {
        self.t_eval = Some(t_eval);
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_t_eval(mut self, t_eval: Option<Vec<Float>>) -> Self {
        self.t_eval = t_eval;
        self
    }

    /// Enable or disable dense output interpolation.
    pub fn dense_output(mut self, dense_output: bool) -> Self {
        self.dense_output = dense_output;
        self
    }

    /// Set the parameters passed to the system functions.
    pub fn p(mut self, p: Vec<Float>) -> Self {
        self.p = p;
        self
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn maybe_p(mut self, p: Option<Vec<Float>>) -> Self {
        if let Some(p) = p {
            self.p = p;
        }
        self
    }
}

impl<F> HamiltonianIvp<'_, F>
where
    F: SeparableHamiltonianSystem,
{
    /// Solve the configured Hamiltonian IVP.
    pub fn solve(self) -> Result<Solution, Error> {
        let config = SymplecticConfig {
            method: self.method,
            step_size: self
                .step_size
                .unwrap_or_else(|| (self.tf - self.t0) / 100.0),
            max_steps: self.max_steps,
            t_eval: self.t_eval,
            dense_output: self.dense_output,
            p: self.p,
        };
        solve_hamiltonian_impl(self.system, self.t0, self.tf, self.q0, self.p0, config)
    }
}
