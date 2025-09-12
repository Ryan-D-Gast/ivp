//! Radau5 - 3-stage, order-5 Radau IIA implicit Runge–Kutta integrator
//!
//! Pragmatic Radau5 integrator:
//! - public `radau5` function, returning `IntegrationResult` and collecting stats
//! - validates `Settings`
//! - simplified Newton iteration with numerical Jacobian via `ODE::jac`
//! - dense output compatible with `SolOut` using a cubic polynomial
//!
//! References:
//! - Hairer & Wanner, Solving Ordinary Differential Equations II (Radau IIA)

use crate::{
	error::Error,
	interpolate::Interpolate,
	matrix::Matrix,
	methods::{
		hinit::hinit,
		result::IntegrationResult,
		settings::{Settings, Tolerance},
	},
	ode::ODE,
	solout::{ControlFlag, SolOut},
	status::Status,
	Float,
};

/// Implicit RK Radau IIA order 5 with adaptive step size and dense output.
pub fn radau5<F, S>(
	f: &F,
	mut x: Float,
	xend: Float,
	y0: &[Float],
	rtol: Tolerance,
	atol: Tolerance,
	mut solout: Option<&mut S>,
	settings: Settings,
) -> Result<IntegrationResult, Vec<Error>>
where
	F: ODE,
	S: SolOut,
{
	// --- Input Validation ---
	let mut errors: Vec<Error> = Vec::new();

	// nmax
	let nmax = settings.nmax.unwrap_or(100_000);
	if nmax == 0 {
		errors.push(Error::NMaxMustBePositive(0));
	}
	// uround
	let uround = match settings.uround {
		Some(u) if (1e-35..1.0).contains(&u) => u,
		Some(u) => {
			errors.push(Error::URoundOutOfRange(u));
			u
		}
		None => 2.3e-16,
	};
	// safety factor
	let safety_factor = match settings.safety_factor {
		Some(s) if s > 1e-4 && s < 1.0 => s,
		Some(s) => {
			errors.push(Error::SafetyFactorOutOfRange(s));
			s
		}
		None => 0.9,
	};
	// Step-size scaling bounds: clamp factor quot in [facc2, facc1]
	let facc1 = settings.scale_min.map(|v| 1.0 / v).unwrap_or(5.0);
	let facc2 = settings.scale_max.map(|v| 1.0 / v).unwrap_or(0.125);
	if facc1 <= 0.0 || !(facc2 < facc1) {
		errors.push(Error::InvalidScaleFactors(
			settings.scale_min.unwrap_or(0.0),
			settings.scale_max.unwrap_or(0.0),
		));
	}
	// settings.beta not used in Radau5 controller; ignore if provided

	let n = y0.len();
	let mut y = y0.to_vec();

	// hmax and hmin
	let hmax = settings.hmax.unwrap_or_else(|| (xend - x).abs());
	let hmin = settings.hmin.unwrap_or(0.0);

	if !errors.is_empty() {
		return Err(errors);
	}

	// Error estimator will use an E1 solve following Hairer-Wanner's Radau5.

	// --- Initialization ---
	let posneg = (xend - x).signum();
	let mut f0 = vec![0.0; n];
	f.ode(x, &y, &mut f0);
	let mut nfev: usize = 1;

	// Initial step size: use provided h0 or compute via hinit (order 5)
	let mut h = if let Some(h0) = settings.h0 {
		h0
	} else {
		let mut f1tmp = vec![0.0; n];
		let mut y1tmp = vec![0.0; n];
		hinit(
			f,
			x,
			&y,
			posneg,
			&f0,
			&mut f1tmp,
			&mut y1tmp,
			5,
			hmax,
			&atol,
			&rtol,
		)
	};
	if h == 0.0 || h.signum() != posneg && posneg != 0.0 {
		return Err(vec![Error::InvalidStepSize(h)]);
	}
	h = h.clamp(-hmax, hmax);

	// Dense output storage: 4 coefficient vectors of length n
	// cont[0]=y_{n+1}, cont[1..3] per Radau5 cubic coefficients
	let mut cont = vec![0.0; n * 4];

	// Initial callback (xold=x)
	if let Some(s) = solout.as_mut() {
		let interp = DenseRadau { cont: &cont, xold: x, h };
		s.solout(x, x, &y, &interp);
	}

	// Working arrays and matrices following the algebraic Radau approach
	// Stage increments Z (added to y at c1, c2, 1), stage transforms F, and stage RHS K
	let mut z1 = vec![0.0; n];
	let mut z2 = vec![0.0; n];
	let mut z3 = vec![0.0; n];
	let mut f1 = vec![0.0; n];
	let mut f2 = vec![0.0; n];
	let mut f3 = vec![0.0; n];
	let mut k1 = vec![0.0; n];
	let mut k2 = vec![0.0; n];
	let mut k3 = vec![0.0; n];
	// Stage state buffers and RHS/solve buffers (reused):
	let mut y1s = vec![0.0; n];
	let mut y2s = vec![0.0; n];
	let mut y3s = vec![0.0; n];
	let mut rhs1 = vec![0.0; n];
	let mut rhs2 = vec![0.0; n];
	let mut rhs3 = vec![0.0; n];
	let mut s1 = vec![0.0; n];
	let mut s2 = vec![0.0; n];
	let mut s3 = vec![0.0; n];
	let mut dF1 = vec![0.0; n];
	let mut dF2 = vec![0.0; n];
	let mut dF3 = vec![0.0; n];
	let mut rhs23 = vec![0.0; 2 * n];
	let mut dF23 = vec![0.0; 2 * n];
	// Error estimator buffers
	let mut tmp = vec![0.0; n];
	let mut f2v = vec![0.0; n];
	let mut contv = vec![0.0; n];
	let mut f1tmp = vec![0.0; n];
	let mut jac = Matrix::zeros(n, n);
	let mut mass = Matrix::identity(n);
	let mut e1 = Matrix::zeros(n, n);
	let mut e2 = Matrix::zeros(2 * n, 2 * n);

	let mut nstep: usize = 0;
	let mut naccpt: usize = 0;
	let mut nrejct: usize = 0;
	let mut status = Status::Success;
	let mut last = false;

	// --- Main loop ---
	while posneg * (xend - x) > 0.0 {
		if nstep > nmax {
			status = Status::NeedLargerNmax;
			break;
		}
		if 0.1 * h.abs() <= x.abs() * uround {
			status = Status::StepSizeTooSmall;
			break;
		}

		if (x + 1.01 * h - xend) * posneg > 0.0 {
			h = xend - x;
			last = true;
		}
		nstep += 1;

		// Jacobian and mass at (x, y)
		f.jac(x, &y, &mut jac);
		f.mass(&mut mass);

		// Build E1 = fac1*M - J and E2 as 2n x 2n block with E2r = alph/h*M - J, E2i = beta/h*M
		let fac1 = U1 / h;
		let alphn = ALPH / h;
		let betan = BETA / h;
		for r in 0..n {
			for c in 0..n {
				e1[(r, c)] = mass[(r, c)] * fac1 - jac[(r, c)];
				let e2r = mass[(r, c)] * alphn - jac[(r, c)];
				let e2i = mass[(r, c)] * betan;
				// Fill 2x2 block
				e2[(r, c)] = e2r; // top-left
				e2[(r, c + n)] = -e2i; // top-right
				e2[(r + n, c)] = e2i; // bottom-left
				e2[(r + n, c + n)] = e2r; // bottom-right
			}
		}

		// Initialize stage increments and transforms
		for i in 0..n { z1[i] = 0.0; z2[i] = 0.0; z3[i] = 0.0; f1[i] = 0.0; f2[i] = 0.0; f3[i] = 0.0; }

		// Stage times (constant during Newton for this step)
	let t1 = x + C1 * h;
	let t2 = x + C2 * h;
		let t3 = x + h;

		// Newton iteration
		let mut newton_ok = false;
		let max_newton = settings.newton_maxiter.unwrap_or(7);
		for _it in 0..max_newton {
			// Stage values
			for i in 0..n { 
                y1s[i] = y[i] + z1[i];
                y2s[i] = y[i] + z2[i];
                y3s[i] = y[i] + z3[i];
            }

			// Evaluate RHS at stages
			f.ode(t1, &y1s, &mut k1);
			f.ode(t2, &y2s, &mut k2);
			f.ode(t3, &y3s, &mut k3);
			nfev += 3;

			// Form RHS via T^{-1} combination of k
			for i in 0..n {
				rhs1[i] = TINV00 * k1[i] + TINV01 * k2[i] + TINV02 * k3[i];
				rhs2[i] = TINV10 * k1[i] + TINV11 * k2[i] + TINV12 * k3[i];
				rhs3[i] = TINV20 * k1[i] + TINV21 * k2[i] + TINV22 * k3[i];
			}

			// Add mass contributions from current F
			for i in 0..n {
				let mut sum1 = 0.0; let mut sum2 = 0.0; let mut sum3 = 0.0;
				for j in 0..n {
					let mij = mass[(i, j)];
					sum1 -= mij * f1[j];
					sum2 -= mij * f2[j];
					sum3 -= mij * f3[j];
				}
				s1[i] = sum1 * fac1;
				s2[i] = sum2 * alphn - sum3 * betan;
				s3[i] = sum3 * alphn + sum2 * betan;
			}
			for i in 0..n { rhs1[i] += s1[i]; rhs2[i] += s2[i]; rhs3[i] += s3[i]; }

			// Solve E1 * dF1 = rhs1
			dF1.copy_from_slice(&rhs1);
			e1.lin_solve_mut(&mut dF1);
			// Solve complex pair via 2n x 2n real system
			for i in 0..n { rhs23[i] = rhs2[i]; rhs23[i + n] = rhs3[i]; }
			dF23.copy_from_slice(&rhs23);
			e2.lin_solve_mut(&mut dF23);
			for i in 0..n { dF2[i] = dF23[i]; dF3[i] = dF23[i + n]; }

			// Update F
			for i in 0..n { f1[i] += dF1[i]; f2[i] += dF2[i]; f3[i] += dF3[i]; }

			// Update Z from F via T matrix
			for i in 0..n {
				z1[i] = T00 * f1[i] + T01 * f2[i] + T02 * f3[i];
				z2[i] = T10 * f1[i] + T11 * f2[i] + T12 * f3[i];
				z3[i] = T20 * f1[i] + f2[i];
			}

			// Convergence based on Newton correction norm
			let mut dyno = 0.0;
			for i in 0..n {
				let sc = atol[i] + rtol[i] * y[i].abs();
				let v1 = dF1[i] / sc;
				let v2 = dF2[i] / sc;
				let v3 = dF3[i] / sc;
				dyno += v1 * v1 + v2 * v2 + v3 * v3;
			}
			dyno = (dyno / (3.0 * n as Float)).sqrt();
			if dyno <= settings.newton_tol.unwrap_or(0.03_f64.min(rtol[0].sqrt())) {
				newton_ok = true;
				break;
			}
		}

		if !newton_ok {
			// Reduce step and retry
			let quot = facc2.max(facc1.min(1.5));
			let hnew = (h / quot).clamp(-hmax, hmax);
			if h.abs() <= hmin || hnew.abs() >= h.abs() {
				status = Status::StepSizeTooSmall;
				break;
			}
			h = hnew;
			nrejct += 1;
			last = false;
			continue;
		}

		// Compute solution at end of step using increment z3
		let mut y1 = y.clone();
		for i in 0..n { y1[i] = y[i] + z3[i]; }

		// Error estimate (Radau5-style via E1 solve with mass matrix):
		// hee_k = dd_k / h; form tmp = hee1*Z1 + hee2*Z2 + hee3*Z3;
		// f2 = M * tmp; cont = f2 + f(x_n, y_n);
		// Solve E1 * cont = cont, with E1 = fac1*M - J, fac1 = U1/h.
		let hee1 = DD1 / h;
		let hee2 = DD2 / h;
		let hee3 = DD3 / h;
		for i in 0..n { tmp[i] = hee1 * z1[i] + hee2 * z2[i] + hee3 * z3[i]; }
		// f2 = M * tmp
		for i in 0..n {
			let mut sum = 0.0;
			for j in 0..n {
				sum += mass[(i, j)] * tmp[j];
			}
			f2v[i] = sum;
		}
		// cont = f(x_n, y_n) + f2
		for i in 0..n { contv[i] = f0[i] + f2v[i]; }
		// Build and solve E1 * cont = cont, E1 = fac1*M - J
		let fac1 = U1 / h;
		for r in 0..n { for c in 0..n { e1[(r, c)] = mass[(r, c)] * fac1 - jac[(r, c)]; } }
		e1.lin_solve_mut(&mut contv);
		let mut err = 0.0;
		for i in 0..n {
			let sk = atol[i] + rtol[i] * y1[i].abs();
			let e = contv[i] / sk;
			err += e * e;
		}
		let mut err = (err / n as Float).sqrt().max(1e-10);
		// Optional refinement: on first or rejected step and large error
		if err >= 1.0 && (naccpt == 0 || nrejct > 0) {
			// cont = y + cont; f1 = f(x, cont)
			let mut ytmp = y.clone();
			for i in 0..n { ytmp[i] += contv[i]; }
			f.ode(x, &ytmp, &mut f1tmp);
			nfev += 1;
			// cont = f1tmp + f2; solve again
			for i in 0..n { contv[i] = f1tmp[i] + f2v[i]; }
			e1.lin_solve_mut(&mut contv);
			// recompute error
			err = 0.0;
			for i in 0..n {
				let sk = atol[i] + rtol[i] * y1[i].abs();
				let e = contv[i] / sk;
				err += e * e;
			}
			err = (err / n as Float).sqrt().max(1e-10);
		}

		// Step size control
		// Classic Radau controller: hnew = h / clamp(err^(1/4)/safety, [facc2, facc1])
		let mut quot = (err.max(1e-10)).powf(0.25) / safety_factor;
		quot = facc2.max(quot.min(facc1));
		let mut hnew = h / quot;

		if err <= 1.0 {
			// accept
			naccpt += 1;

			// Dense output coefficients per Radau5 (cubic at right endpoint)
			// Compute stage increments Z_i = z_i - y (using left-end y)
			let c1m1 = C1M1;
			let c2m1 = C2M1;
			let c1mc2 = C1MC2;
			for i in 0..n {
				// cont0 = y_{n+1}
				cont[0 * n + i] = y1[i];
				// cont1 = (Z2 - Z3)/(c2-1)
				let cont1i = (z2[i] - z3[i]) / c2m1;
				// ak = (Z1 - Z2)/(c1 - c2)
					let ak = (z1[i] - z2[i]) / c1mc2;
				// acont3 = (ak - Z1/c1)/c2
					let acont3 = (ak - (z1[i] / C1)) / C2;
				// cont2 = (ak - cont1)/(c1-1)
				let cont2i = (ak - cont1i) / c1m1;
				// cont3 = cont2 - acont3
				let cont3i = cont2i - acont3;
				cont[1 * n + i] = cont1i;
				cont[2 * n + i] = cont2i;
				cont[3 * n + i] = cont3i;
			}

			// Update state
			y.copy_from_slice(&y1);
			let xold = x;
			x += h;
			// update derivative at new point for next step
			f.ode(x, &y, &mut f0);
			nfev += 1;

			// Callback
			if let Some(ref mut s) = solout {
				match s.solout(xold, x, &y, &DenseRadau { cont: &cont, xold, h }) {
					ControlFlag::Continue => {}
					ControlFlag::Interrupt => {
						status = Status::Interrupted;
						break;
					}
					ControlFlag::ModifiedSolution(nx, ny) => {
						x = nx;
						y.copy_from_slice(&ny);
						f.ode(x, &y, &mut f0);
						nfev += 1;
					}
				}
			}

			if last {
				h = hnew;
				status = Status::Success;
				break;
			}

			// step size limits
			if hnew.abs() > hmax {
				hnew = hmax.copysign(posneg);
			}
			h = hnew;
		} else {
			// reject
			let mut quot_rej = (err.max(1e-10)).powf(0.25) / safety_factor;
			quot_rej = facc2.max(quot_rej.min(facc1));
			let htry = (h / quot_rej).clamp(-hmax, hmax);
			if h.abs() <= hmin || htry.abs() >= h.abs() {
				status = Status::StepSizeTooSmall;
				break;
			}
			h = htry;
			nrejct += 1;
			last = false;
		}
	}

	Ok(IntegrationResult {
		x,
		y,
		h,
		status,
		nfev,
		nstep,
		naccpt,
		nrejct,
	})
}

/// Dense output for Radau5 compatible with `SolOut`.
/// Cubic polynomial based at the right endpoint using coefficients in `cont`.
struct DenseRadau<'a> {
	cont: &'a [Float], // [c0(n)=y_{n+1}, c1(n), c2(n), c3(n)]
	xold: Float,
	h: Float,
}

impl<'a> Interpolate for DenseRadau<'a> {
	fn interpolate(&self, xi: Float, yi: &mut [Float]) {
		let n = self.cont.len() / 4;
		// s = (t - t_{n+1}) / h = theta - 1, but since q = (xi - (xold + h)) / h = (xi - x) / h
		// and x = xold + h, so s = q
		let s = (xi - (self.xold + self.h)) / self.h;
		let c0 = &self.cont[0 * n..1 * n];
		let c1 = &self.cont[1 * n..2 * n];
		let c2 = &self.cont[2 * n..3 * n];
		let c3 = &self.cont[3 * n..4 * n];
		for i in 0..n {
			yi[i] = c0[i] + s * (c1[i] + (s - C2M1) * (c2[i] + (s - C1M1) * c3[i]));
		}
	}

	fn get_cont(&self) -> (Vec<Float>, Float, Float) {
		(self.cont.to_vec(), self.xold, self.h)
	}
}

// Nodes (abscissae) in [0,1]
const C1: Float = 0.155_051_025_721_682_2;
const C2: Float = 0.644_948_974_278_317_8;
const C1M1: Float = C1 - 1.0;
const C2M1: Float = C2 - 1.0;
const C1MC2: Float = C1 - C2;

// Error estimation and splitting constants
const DD1: Float = -10.048_809_399_827_416;
const DD2: Float = 1.382_142_733_160_749;
const DD3: Float = -0.333_333_333_333_333_3;
const U1: Float = 3.637_834_252_744_496; // real system coefficient
const ALPH: Float = 2.681_082_873_627_752_3;
const BETA: Float = 3.050_430_199_247_410_5;

// Transformation matrix T (3x3) constants
const T00: Float = 9.123_239_487_089_295E-2;
const T01: Float = -1.412_552_950_209_542E-1;
const T02: Float = -3.002_919_410_514_742_4E-2;
const T10: Float = 2.417_179_327_071_07E-1;
const T11: Float = 2.041_293_522_937_999_4E-1;
const T12: Float = 3.829_421_127_572_619E-1;
const T20: Float = 9.660_481_826_150_93E-1;

// Inverse transformation matrix T^{-1} constants
const TINV00: Float = 4.325_579_890_063_155;
const TINV01: Float = 3.391_992_518_158_098_4E-1;
const TINV02: Float = 5.417_705_399_358_749E-1;
const TINV10: Float = -4.178_718_591_551_905;
const TINV11: Float = -3.276_828_207_610_623_7E-1;
const TINV12: Float = 4.766_235_545_005_504_4E-1;
const TINV20: Float = -5.028_726_349_457_868E-1;
const TINV21: Float = 2.571_926_949_855_605;
const TINV22: Float = -5.960_392_048_282_249E-1;
