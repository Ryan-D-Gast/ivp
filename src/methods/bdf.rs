//! BDF — variable order (1..5) Backward Differentiation Formula solver.

use crate::{
    error::{Error, ConfigError},
    interpolate::Interpolate,
    matrix::{lin_solve, lu_decomp, Matrix, MatrixStorage},
    methods::{hinit, Evals, IntegrationResult, Steps, Tolerance},
    ivp::IVP,
    solout::{ControlFlag, SolOut},
    status::Status,
    Float,
};
use bon::Builder;

const MAX_ORDER: usize = 5;
const MIN_FACTOR: Float = 0.2;
const MAX_FACTOR: Float = 10.0;
const SAFETY_DEFAULT: Float = 0.9;
const BDF_COEFFS_PER_STATE: usize = MAX_ORDER + 2;

// Fixed coefficients
const KAPPA: [Float; MAX_ORDER + 1] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];

/// BDF — variable order (1..5) Backward Differentiation Formula solver with configuration.
#[derive(Builder, Clone, Debug)]
pub struct BDF {
    /// Maximum number of steps (default: 100_000)
    #[builder(default = 100_000)]
    pub max_steps: usize,
    /// Maximum step size (default: None = |xend - x|)
    pub max_step: Option<Float>,
    /// Minimum step size (default: None = 0.0)
    pub min_step: Option<Float>,
    /// Maximum Newton iterations per step (default: 4)
    #[builder(default = 4)]
    pub newton_maxiter: usize,
    /// Newton convergence tolerance (default: None = derived from tolerances)
    pub newton_tol: Option<Float>,
    /// Jacobian matrix storage (default: Full)
    #[builder(default = MatrixStorage::Full)]
    pub jac_storage: MatrixStorage,
    /// Initial step size (default: None = automatic)
    pub first_step: Option<Float>,
}

impl Default for BDF {
    fn default() -> Self {
        Self {
            max_steps: 100_000,
            max_step: None,
            min_step: None,
            newton_maxiter: 4,
            newton_tol: None,
            jac_storage: MatrixStorage::Full,
            first_step: None,
        }
    }
}

impl BDF {
    /// Solve an initial value problem using the variable-order BDF(1-5) method.
    ///
    /// This function integrates the autonomous system `y' = f(x, y)` from `x` to
    /// `xend`, advancing the provided state buffer `y` in-place. It uses a
    /// variable-order Backward Differentiation Formula method with adaptive
    /// step-size control and optional dense output.
    ///
    /// # Arguments
    ///
    /// ## Defining the Problem
    /// - `f`: Right‑hand side implementing `IVP`.
    /// - `x`: Initial independent variable value.
    /// - `xend`: Final independent variable value.
    /// - `y`: Mutable slice containing the initial state; on success contains the
    ///   state at the final time.
    /// - `rtol`, `atol`: Relative and absolute tolerances (see [`Tolerance`]).
    ///
    /// ## Output Control
    /// - `solout`: Optional mutable reference to a `SolOut` callback used for
    ///   intermediate output. If provided, the callback may receive a dense
    ///   interpolant for efficient interpolation within accepted steps.
    ///
    /// Solver settings are configured via the `BDF` struct fields.
    ///
    /// # Returns
    /// A `Result` with `IntegrationResult` on success or an `Error` if validation fails.
    pub fn solve<F, S>(
        &self,
        f: &F,
        x: &mut Float,
        y: &mut [Float],
        xend: Float,
        rtol: Tolerance,
        atol: Tolerance,
        mut solout: Option<&mut S>,
    ) -> Result<IntegrationResult, Error>
    where
        F: IVP,
        S: SolOut,
    {
        let n = y.len();
        if n == 0 {
            return Ok(IntegrationResult::new(
                0.0,
                Status::Success,
                Evals::new(),
                Steps::new(),
            ));
        }
        
        // Convert tolerances to per-component vectors and validate non-negativity/length
        for i in 0..n {
            if rtol[i] < 0.0 {
                return Err(Error::Config(ConfigError::NegativeTolerance {
                    kind: "relative",
                    index: i,
                    value: rtol[i],
                }));
            }
            if atol[i] < 0.0 {
                return Err(Error::Config(ConfigError::NegativeTolerance {
                    kind: "absolute",
                    index: i,
                    value: atol[i],
                }));
            }
        }

        let nmax = self.max_steps;
        if nmax == 0 {
            return Err(Error::Config(ConfigError::MustBePositive {
                parameter: "max_steps",
                value: nmax,
            }));
        }

        let diff = xend - *x;
        let direction = diff.signum();

        let hmax = self.max_step.unwrap_or_else(|| (xend - *x).abs()).abs();
        let hmin = self.min_step.unwrap_or(0.0).abs();

        let mut evals = Evals::new();
        let mut steps = Steps::new();

        // Workspace vectors
        let mut f0 = vec![0.0; n];
        f.ode(*x, y, &mut f0);
        evals.ode += 1;

        let mut jac = Matrix::from_storage(n, n, self.jac_storage.clone());
        f.jac(*x, y, &mut jac);
        evals.jac += 1;

        // Precompute gamma, alpha, error_const arrays
        let mut gamma = [0.0; MAX_ORDER + 1];
        for k in 1..=MAX_ORDER {
            gamma[k] = gamma[k - 1] + 1.0 / k as Float;
        }
        let mut alpha = [0.0; MAX_ORDER + 1];
        for k in 0..=MAX_ORDER {
            alpha[k] = (1.0 - KAPPA[k]) * gamma[k];
        }
        let mut error_const = [0.0; MAX_ORDER + 1];
        for k in 0..=MAX_ORDER {
            error_const[k] = KAPPA[k] * gamma[k] + 1.0 / (k as Float + 1.0);
        }

        let rtol_min = rtol
            .iter(n)
            .fold(Float::INFINITY, Float::min)
            .max(Float::EPSILON);
        let mut newton_tol_val = self.newton_tol.unwrap_or_else(|| {
            let eps_term = 10.0 * Float::EPSILON / rtol_min;
            let sqrt_term = rtol_min.sqrt().min(0.03);
            eps_term.max(sqrt_term)
        });
        if newton_tol_val <= 0.0 {
            newton_tol_val = 1e-9;
        }
        let newton_maxiter_val = self.newton_maxiter.max(1);

        // Initial step size selection
        let mut h_abs = if let Some(h) = self.first_step {
            // Auto-correct the sign - just use absolute value
            // since h_abs is always positive and direction is applied later
            if h == 0.0 {
                return Err(Error::Config(ConfigError::InvalidStepSize {
                    value: h,
                    expected_sign: direction,
                }));
            }
            h.abs()
        } else {
            let mut f1 = vec![0.0; n];
            let mut y1 = vec![0.0; n];
            let guess = hinit(
                f, *x, y, direction, &f0, &mut f1, &mut y1, 1, hmax, &atol, &rtol,
            );
            // Ensure x + h isn't larger than xend
            let diff = xend - *x;
            let max_h = diff.abs();
            let guess = if guess.abs() > max_h {
                max_h * direction
            } else {
                guess
            };
            guess.abs()
        };

        h_abs = h_abs.min(hmax.max(Float::MIN_POSITIVE));
        let mut current_h = h_abs;

        // Difference arrays (order+3) x n
        let mut d = vec![vec![0.0; n]; MAX_ORDER + 3];
        d[0].copy_from_slice(y);
        for i in 0..n {
            d[1][i] = f0[i] * current_h * direction;
        }

        let mut order = 1usize;
        let mut n_equal_steps = 0usize;
        let status;

        let mut psi = vec![0.0; n];
        let mut scale = vec![0.0; n];
        let mut y_predict = vec![0.0; n];
        let mut y_new = vec![0.0; n];
        let mut delta = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        let mut scratch_change = vec![vec![0.0; n]; MAX_ORDER + 1];
        let mut lu_matrix = Matrix::zeros(n, n);
        let mut pivot = vec![0usize; n];

        // Dense output coefficients for current step
        const CONT_BLOCK: usize = BDF_COEFFS_PER_STATE; // D0, D1..D5, order marker
        let mut cont = vec![0.0; n * CONT_BLOCK];
        // Initial callback
        if let Some(sol) = solout.as_mut() {
            match sol.solout::<DenseBdf15>(*x, x, y, None) {
                ControlFlag::Continue => {}
                ControlFlag::Interrupt => {
                    return Ok(IntegrationResult::new(
                        direction * current_h,
                        Status::UserInterrupt,
                        evals,
                        steps,
                    ));
                }
                ControlFlag::ModifiedSolution => {
                    // Update derivatives at new (x, y).
                    f.ode(*x, y, &mut f0);
                    evals.ode += 1;
                    d[0].copy_from_slice(y);
                    for i in 0..n {
                        d[1][i] = f0[i] * current_h * direction;
                    }
                    for k in 2..d.len() {
                        d[k].fill(0.0);
                    }
                    order = 1;
                    n_equal_steps = 0;
                    f.jac(*x, y, &mut jac);
                    evals.jac += 1;
                }
                ControlFlag::XOut(_) => {}
            }
        }

        'main_loop: loop {
            if steps.total >= nmax {
                status = Status::NeedLargerNMax;
                break;
            }

            if current_h < Float::MIN_POSITIVE {
                status = Status::StepSizeTooSmall;
                break;
            }

            let mut h_try = current_h;
            if h_try > hmax {
                let factor = hmax / h_try;
                change_d(&mut d, order, factor, &mut scratch_change);
                h_try = hmax;
                current_h = h_try;
                n_equal_steps = 0;
            }
            if h_try < hmin && hmin > 0.0 {
                let factor = (hmin / h_try).max(1.0);
                change_d(&mut d, order, factor, &mut scratch_change);
                h_try = hmin;
                current_h = h_try;
                n_equal_steps = 0;
            }

            let mut h_signed = direction * h_try;
            let x_start = *x;
            let mut x_new = *x + h_signed;
            if direction * (x_new - xend) > 0.0 {
                let step_to_end = (xend - *x).abs();
                if step_to_end == 0.0 {
                    status = Status::Success;
                    break;
                }
                let factor = step_to_end / h_try;
                change_d(&mut d, order, factor, &mut scratch_change);
                current_h *= factor;
                h_try = current_h;
                h_signed = direction * h_try;
                x_new = *x + h_signed;
                n_equal_steps = 0;
            }

            // Step size guard against stagnation
            if (*x + 0.1 * h_signed.abs()) == *x {
                status = Status::StepSizeTooSmall;
                break;
            }

            steps.total += 1;

            // Predictor: y_{n+1} ≈ sum_{i=0}^order D[i]
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..=order {
                    sum += d[k][i];
                }
                y_predict[i] = sum;
            }

            // Scale and psi
            for i in 0..n {
                scale[i] = atol[i] + rtol[i] * y_predict[i].abs();
                if scale[i] == 0.0 {
                    scale[i] = Float::EPSILON;
                }
            }
            for i in 0..n {
                let mut s = 0.0;
                for j in 1..=order {
                    s += gamma[j] * d[j][i];
                }
                psi[i] = s / alpha[order];
            }

            // Build and factor LU of (I - c*J)
            let c = h_signed / alpha[order];
            for r in 0..n {
                for c_idx in 0..n {
                    lu_matrix[(r, c_idx)] = -c * jac[(r, c_idx)];
                }
                lu_matrix[(r, r)] += 1.0;
            }
            evals.lu += 1;
            match lu_decomp(&mut lu_matrix, &mut pivot) {
                Ok(()) => {}
                Err(_) => {
                    let factor = 0.5;
                    change_d(&mut d, order, factor, &mut scratch_change);
                    current_h *= factor;
                    n_equal_steps = 0;
                    steps.rejected += 1;
                    continue 'main_loop;
                }
            }

            // Simplified Newton iterations
            y_new.copy_from_slice(&y_predict);
            delta.fill(0.0);
            let mut converged = false;
            let mut dy_norm_prev: Option<Float> = None;
            let mut iters = 0usize;
            while iters < newton_maxiter_val {
                f.ode(x_new, &y_new, &mut rhs);
                evals.ode += 1;
                for i in 0..n {
                    rhs[i] = c * rhs[i] - psi[i] - delta[i];
                }
                lin_solve(&lu_matrix, &mut rhs, &pivot);

                let dy_norm = weighted_rms_scaled(&rhs, &scale);
                let mut rate_condition = false;
                if let Some(prev) = dy_norm_prev {
                    if prev > 0.0 {
                        let rate = dy_norm / prev;
                        if rate >= 1.0 {
                            rate_condition = true;
                        } else {
                            let remaining = (newton_maxiter_val - iters) as Float;
                            let estimate = rate.powf(remaining) / (1.0 - rate) * dy_norm;
                            if estimate > newton_tol_val {
                                rate_condition = true;
                            }
                            if rate_condition {
                                // will trigger jacobian refresh below
                            }
                        }
                    }
                }

                for i in 0..n {
                    y_new[i] += rhs[i];
                    delta[i] += rhs[i];
                }

                if dy_norm == 0.0 {
                    converged = true;
                    break;
                }
                if let Some(prev) = dy_norm_prev {
                    if prev > 0.0 {
                        let rate = dy_norm / prev;
                        if rate < 1.0 {
                            let estimate = rate / (1.0 - rate) * dy_norm;
                            if estimate < newton_tol_val {
                                converged = true;
                                break;
                            }
                        }
                    }
                }

                if rate_condition {
                    break;
                }

                dy_norm_prev = Some(dy_norm);
                iters += 1;
            }
            if !converged {
                // Refresh Jacobian and retry step unless already tried
                f.jac(x_new, &y_predict, &mut jac);
                evals.jac += 1;
                change_d(&mut d, order, 0.5, &mut scratch_change);
                current_h *= 0.5;
                n_equal_steps = 0;
                steps.rejected += 1;
                continue;
            }

            let safety = SAFETY_DEFAULT * (2.0 * newton_maxiter_val as Float + 1.0)
                / (2.0 * newton_maxiter_val as Float + (iters + 1) as Float);

            for i in 0..n {
                scale[i] = atol[i] + rtol[i] * y_new[i].abs();
                if scale[i] == 0.0 {
                    scale[i] = Float::EPSILON;
                }
            }

            let error_norm = {
                for i in 0..n {
                    rhs[i] = error_const[order] * delta[i];
                    if scale[i] == 0.0 {
                        scale[i] = Float::EPSILON;
                    }
                }
                weighted_rms_scaled(&rhs, &scale)
            };

            if error_norm > 1.0 {
                let mut factor = safety * error_norm.powf(-1.0 / (order as Float + 1.0));
                factor = factor.max(MIN_FACTOR);
                change_d(&mut d, order, factor, &mut scratch_change);
                current_h *= factor;
                *x = *x;
                n_equal_steps = 0;
                steps.rejected += 1;
                continue;
            }

            steps.accepted += 1;
            n_equal_steps += 1;
            *x = x_new;
            y.copy_from_slice(&y_new);
            for i in 0..n {
                d[order + 2][i] = delta[i] - d[order + 1][i];
                d[order + 1][i] = delta[i];
            }
            for k in (0..=order).rev() {
                for i in 0..n {
                    d[k][i] += d[k + 1][i];
                }
            }

            // Prepare dense coefficients
            for i in 0..n {
                let base = i * CONT_BLOCK;
                cont[base] = d[0][i];
                for k in 0..MAX_ORDER {
                    let coeff = if k + 1 <= order { d[k + 1][i] } else { 0.0 };
                    cont[base + 1 + k] = coeff;
                }
                cont[base + CONT_BLOCK - 1] = order as Float;
            }

            // Callback
            if let Some(sol) = solout.as_mut() {
                let interp = DenseBdf15::new(cont.as_ptr(), cont.len(), x_start, h_signed);
                match sol.solout::<DenseBdf15>(*x - h_signed, x, y, Some(&interp)) {
                    ControlFlag::Continue => {}
                    ControlFlag::Interrupt => {
                        status = Status::UserInterrupt;
                        break;
                    }
                    ControlFlag::ModifiedSolution => {
                        // Update derivatives at new (x, y).
                        f.ode(*x, y, &mut f0);
                        evals.ode += 1;
                        d[0].copy_from_slice(y);
                        for i in 0..n {
                            d[1][i] = f0[i] * current_h * direction;
                        }
                        for k in 2..d.len() {
                            d[k].fill(0.0);
                        }
                        order = 1;
                        n_equal_steps = 0;
                        f.jac(*x, y, &mut jac);
                        evals.jac += 1;
                    }
                    ControlFlag::XOut(_) => {}
                }
            }

            if direction * (*x - xend) >= 0.0 {
                status = Status::Success;
                break;
            }

            // Order/step-size adaptation when enough equal steps
            if n_equal_steps >= order + 1 {
                let mut err_m = Float::INFINITY;
                let mut err_p = Float::INFINITY;
                if order > 1 {
                    for i in 0..n {
                        rhs[i] = error_const[order - 1] * d[order][i];
                    }
                    err_m = weighted_rms_scaled(&rhs, &scale);
                }
                if order < MAX_ORDER {
                    for i in 0..n {
                        rhs[i] = error_const[order + 1] * d[order + 2][i];
                    }
                    err_p = weighted_rms_scaled(&rhs, &scale);
                }

                // SciPy approach: compute factors = error_norms ** (-1 / (order + k))
                // When error_norm is 0, this gives infinity, which gets capped by MAX_FACTOR
                let errors = [err_m, error_norm, err_p];
                let mut factors = [0.0; 3];
                for (idx, err) in errors.iter().enumerate() {
                    let exponent = -1.0 / (order as Float + idx as Float);
                    // 0.0.powf(negative) = inf in Rust, matching SciPy behavior
                    factors[idx] = err.powf(exponent);
                }
                
                // Find best order change: argmax(factors) - 1
                let (best_idx, _) = factors
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((1, &1.0));

                let mut new_order = order;
                if best_idx == 0 && order > 1 {
                    new_order -= 1;
                } else if best_idx == 2 && order < MAX_ORDER {
                    new_order += 1;
                }

                // SciPy: factor = min(MAX_FACTOR, safety * max(factors))
                let max_factor = factors.iter().cloned().fold(0.0, Float::max);
                let step_factor = (safety * max_factor).min(MAX_FACTOR);
                change_d(&mut d, new_order, step_factor, &mut scratch_change);
                current_h *= step_factor;
                order = new_order;
                n_equal_steps = 0;
            } else {
            }

            // Update Jacobian for next step
            f.jac(*x, y, &mut jac);
            evals.jac += 1;
        }

        Ok(IntegrationResult::new(
            direction * current_h,
            status,
            evals,
            steps,
        ))
    }

    /// Dense-output evaluation for BDF(1..5).
    pub fn interpolate(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
        if h == 0.0 {
            return;
        }
        const BLOCK: usize = BDF_COEFFS_PER_STATE;
        let n = yi.len();
        
        // Handle empty state vector case
        if n == 0 {
            return;
        }
        
        debug_assert_eq!(cont.len(), n * BLOCK);

        let order = cont[BLOCK - 1].round().clamp(1.0, MAX_ORDER as Float) as usize;
        let x_new = xold + h;

        let mut x_factors = [0.0; MAX_ORDER];
        let mut p = [0.0; MAX_ORDER];
        for k in 0..order {
            let denom = h * (k as Float + 1.0);
            let t_shift = x_new - h * k as Float;
            x_factors[k] = (xi - t_shift) / denom;
            if k == 0 {
                p[0] = x_factors[0];
            } else {
                p[k] = p[k - 1] * x_factors[k];
            }
        }

        for i in 0..n {
            let base = i * BLOCK;
            let mut sum = cont[base];
            for k in 0..order {
                sum += cont[base + 1 + k] * p[k];
            }
            yi[i] = sum;
        }
    }
}

fn weighted_rms_scaled(values: &[Float], scale: &[Float]) -> Float {
    let mut sum = 0.0;
    for (v, s) in values.iter().zip(scale.iter()) {
        let denom = if *s == 0.0 { Float::EPSILON } else { *s };
        let ratio = v / denom;
        sum += ratio * ratio;
    }
    (sum / values.len() as Float).sqrt()
}

fn change_d(d: &mut [Vec<Float>], order: usize, factor: Float, scratch: &mut [Vec<Float>]) {
    if factor == 1.0 {
        return;
    }
    let order = order.min(MAX_ORDER);
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    let ru = matmul(&r, &u);
    for (row_idx, scratch_row) in scratch.iter_mut().enumerate().take(order + 1) {
        scratch_row.fill(0.0);
        for k in 0..=order {
            let coeff = ru[k][row_idx];
            if coeff == 0.0 {
                continue;
            }
            for (s, dval) in scratch_row.iter_mut().zip(d[k].iter()) {
                *s += coeff * dval;
            }
        }
    }
    for i in 0..=order {
        d[i].copy_from_slice(&scratch[i]);
    }
}

fn compute_r(order: usize, factor: Float) -> Vec<Vec<Float>> {
    let size = order + 1;
    let mut m = vec![vec![0.0; size]; size];
    for j in 0..size {
        m[0][j] = 1.0;
    }
    for i in 1..size {
        for j in 1..size {
            m[i][j] = (i as Float - 1.0 - factor * j as Float) / i as Float;
        }
    }
    let mut r = vec![vec![0.0; size]; size];
    r[0].clone_from_slice(&m[0]);
    for i in 1..size {
        for j in 0..size {
            r[i][j] = r[i - 1][j] * m[i][j];
        }
    }
    r
}

fn matmul(a: &[Vec<Float>], b: &[Vec<Float>]) -> Vec<Vec<Float>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();
    let mut res = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for k in 0..inner {
            let coeff = a[i][k];
            if coeff == 0.0 {
                continue;
            }
            for j in 0..cols {
                res[i][j] += coeff * b[k][j];
            }
        }
    }
    res
}

struct DenseBdf15 {
    cont_ptr: *const Float,
    cont_len: usize,
    xold: Float,
    h: Float,
}

impl DenseBdf15 {
    fn new(cont_ptr: *const Float, cont_len: usize, xold: Float, h: Float) -> Self {
        Self {
            cont_ptr,
            cont_len,
            xold,
            h,
        }
    }
}

impl Interpolate for DenseBdf15 {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        unsafe {
            let cont = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            BDF::interpolate(xi, yi, cont, self.xold, self.h);
        }
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        unsafe {
            let cont = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            (cont.to_vec(), self.xold, self.h)
        }
    }
}
