#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! Pure-Rust LSODA translation for dense Jacobian storage.
//!
//! This module ports the core Adams/BDF automatic-switching logic from the
//! `lsoda.c` reference into Rust. The current scope intentionally covers the
//! dense Jacobian modes only: Adams uses functional iteration and BDF uses a
//! dense chord iteration with Jacobians supplied through
//! [`FirstOrderSystem::jac`]. Banded Jacobian storage is rejected explicitly.

use bon::Builder;

use crate::{
    Float,
    dense::StepInterpolant,
    error::{ConfigError, Error},
    ivp::FirstOrderSystem,
    matrix::{Matrix, MatrixStorage},
    methods::{Evals, IntegrationResult, Steps, Tolerance},
    solout::{ControlFlag, SolOut},
    solve::JacobianSource,
    status::Status,
};

const MAX_STEPS_DEFAULT: usize = 100_000;
const MXORDN_DEFAULT: usize = 12;
const MXORDS_DEFAULT: usize = 5;
const MAXCOR_DEFAULT: usize = 3;
const MSBP_DEFAULT: usize = 20;
const MXNCF_DEFAULT: usize = 10;
const ELCO_ROWS: usize = 13;
const ELCO_COLS: usize = 12;
const TESCO_ROWS: usize = 3;
const LSODA_SNAPSHOT_HEADER: usize = 6;
const SM1: [Float; 12] = [
    0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15, 0.10, 0.075, 0.050, 0.025,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MethodFamily {
    Adams = 1,
    Bdf = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorrectorStatus {
    Converged,
    Retry,
    NoConvergence,
}

#[derive(Clone, Copy, Debug)]
struct CorrectorOutput {
    status: CorrectorStatus,
    m: usize,
    del: Float,
}

/// Pure-Rust LSODA solver configuration.
#[derive(Builder, Clone, Debug)]
pub struct LSODA {
    /// Maximum number of accepted steps.
    #[builder(default = MAX_STEPS_DEFAULT)]
    pub max_steps: usize,
    /// Maximum step size.
    pub max_step: Option<Float>,
    /// Minimum step size.
    pub min_step: Option<Float>,
    /// Initial step size.
    pub first_step: Option<Float>,
    /// Jacobian storage used in the stiff branch.
    #[builder(default = MatrixStorage::Full)]
    pub jac_storage: MatrixStorage,
    /// Select whether LSODA uses its internal dense finite-difference Jacobian
    /// logic or calls the system's `jac(...)` implementation directly.
    #[builder(default = JacobianSource::Auto)]
    pub jacobian_source: JacobianSource,
    /// Maximum nonstiff (Adams) order.
    #[builder(default = MXORDN_DEFAULT)]
    pub mxordn: usize,
    /// Maximum stiff (BDF) order.
    #[builder(default = MXORDS_DEFAULT)]
    pub mxords: usize,
    /// Maximum corrector iterations.
    #[builder(default = MAXCOR_DEFAULT)]
    pub maxcor: usize,
    /// Steps between forced Jacobian refresh checks.
    #[builder(default = MSBP_DEFAULT)]
    pub msbp: usize,
    /// Maximum consecutive convergence failures before aborting.
    #[builder(default = MXNCF_DEFAULT)]
    pub mxncf: usize,
}

impl Default for LSODA {
    fn default() -> Self {
        Self {
            max_steps: MAX_STEPS_DEFAULT,
            max_step: None,
            min_step: None,
            first_step: None,
            jac_storage: MatrixStorage::Full,
            jacobian_source: JacobianSource::Auto,
            mxordn: MXORDN_DEFAULT,
            mxords: MXORDS_DEFAULT,
            maxcor: MAXCOR_DEFAULT,
            msbp: MSBP_DEFAULT,
            mxncf: MXNCF_DEFAULT,
        }
    }
}

#[derive(Clone, Debug)]
struct Common {
    n: usize,
    tn: Float,
    h: Float,
    hu: Float,
    hold: Float,
    hmin: Float,
    hmxi: Float,
    uround: Float,
    tsw: Float,
    meth: MethodFamily,
    mused: usize,
    miter: usize,
    jtyp: usize,
    nq: usize,
    l: usize,
    maxord: usize,
    mxordn: usize,
    mxords: usize,
    maxcor: usize,
    msbp: usize,
    mxncf: usize,
    jstart: isize,
    ipup: usize,
    nslp: usize,
    icount: isize,
    irflag: usize,
    ialth: usize,
    lmax: usize,
    nqnyh: usize,
    nst: usize,
    nfe: usize,
    nje: usize,
    nlu: usize,
    nqu: usize,
    kflag: isize,
    jcur: usize,
    ierpj: isize,
    iersl: isize,
    icf: usize,
    ccmax: Float,
    conv_rate: Float,
    el0: Float,
    conit: Float,
    rc: Float,
    rmax: Float,
    pdest: Float,
    pdlast: Float,
    pdnorm: Float,
    ratio: Float,
    rejected: usize,
    total_steps: usize,
    el: [Float; ELCO_ROWS],
    elco: [Float; ELCO_ROWS * ELCO_COLS],
    tesco: [Float; TESCO_ROWS * ELCO_COLS],
    cm1: [Float; ELCO_COLS],
    cm2: [Float; MXORDS_DEFAULT],
}

struct Work {
    yh: Vec<Float>,
    ewt: Vec<Float>,
    savf: Vec<Float>,
    ftem: Vec<Float>,
    acor: Vec<Float>,
    wm: Vec<Float>,
    ipiv: Vec<usize>,
    jac: Matrix,
}

impl Work {
    fn new(n: usize, lmax: usize) -> Self {
        Self {
            yh: vec![0.0; lmax * n],
            ewt: vec![0.0; n],
            savf: vec![0.0; n],
            ftem: vec![0.0; n],
            acor: vec![0.0; n],
            wm: vec![0.0; 2 + n * n],
            ipiv: vec![0; n],
            jac: Matrix::zeros(n, n),
        }
    }
}

impl LSODA {
    /// Solve a first-order IVP with automatic Adams/BDF switching.
    pub fn solve<F, S>(
        &self,
        system: &F,
        x0: Float,
        y0: &[Float],
        xend: Float,
        rtol: Tolerance,
        atol: Tolerance,
        mut solout: Option<&mut S>,
    ) -> Result<IntegrationResult, Error>
    where
        F: FirstOrderSystem,
        S: SolOut,
    {
        validate_common(self, y0.len(), x0, xend, &rtol, &atol)?;

        if y0.is_empty() {
            return Ok(IntegrationResult::new(
                0.0,
                Status::Success,
                Evals::new(),
                Steps::new(),
            ));
        }

        let n = y0.len();
        let direction = (xend - x0).signum();
        let hmax = self.max_step.unwrap_or_else(|| (xend - x0).abs()).abs();
        let hmin = self.min_step.unwrap_or(0.0).abs();
        let hmxi = if hmax > 0.0 { 1.0 / hmax } else { 0.0 };
        let lmax = self.mxordn.max(self.mxords) + 1;

        let mut common = Common {
            n,
            tn: x0,
            h: 0.0,
            hu: 0.0,
            hold: 0.0,
            hmin,
            hmxi,
            uround: Float::EPSILON,
            tsw: x0,
            meth: MethodFamily::Adams,
            mused: 0,
            miter: 0,
            jtyp: match self.jacobian_source {
                JacobianSource::UserProvided => 1,
                JacobianSource::Auto | JacobianSource::InternalFiniteDifference => 2,
            },
            nq: 1,
            l: 2,
            maxord: self.mxordn.min(MXORDN_DEFAULT),
            mxordn: self.mxordn.min(MXORDN_DEFAULT),
            mxords: self.mxords.min(MXORDS_DEFAULT),
            maxcor: self.maxcor.max(1),
            msbp: self.msbp.max(1),
            mxncf: self.mxncf.max(1),
            jstart: 0,
            ipup: 0,
            nslp: 0,
            icount: 20,
            irflag: 0,
            ialth: 2,
            lmax,
            nqnyh: n,
            nst: 0,
            nfe: 0,
            nje: 0,
            nlu: 0,
            nqu: 0,
            kflag: 0,
            jcur: 0,
            ierpj: 0,
            iersl: 0,
            icf: 0,
            ccmax: 0.3,
            conv_rate: 0.7,
            el0: 1.0,
            conit: 0.0,
            rc: 0.0,
            rmax: 0.0,
            pdest: 0.0,
            pdlast: 0.0,
            pdnorm: 0.0,
            ratio: 5.0,
            rejected: 0,
            total_steps: 0,
            el: [0.0; ELCO_ROWS],
            elco: [0.0; ELCO_ROWS * ELCO_COLS],
            tesco: [0.0; TESCO_ROWS * ELCO_COLS],
            cm1: [0.0; ELCO_COLS],
            cm2: [0.0; MXORDS_DEFAULT],
        };

        let mut work = Work::new(n, lmax);
        let mut y = y0.to_vec();

        system.derivative(x0, &y, &mut work.savf);
        common.nfe += 1;
        work.yh[..n].copy_from_slice(&y);

        ewset(&rtol, &atol, &work.yh[..n], &mut work.ewt);
        invert_weights(&mut work.ewt)?;

        common.h = if let Some(h0) = self.first_step {
            h0
        } else {
            initial_step_size(x0, xend, &y, &work.savf, &work.ewt, &rtol, &atol)
        };
        if common.h == 0.0 || common.h.signum() != direction {
            return Err(Error::Config(ConfigError::InvalidStepSize {
                value: common.h,
                expected_sign: direction,
            }));
        }
        if common.hmxi != 0.0 {
            common.h /= (common.h.abs() * common.hmxi).max(1.0);
        }
        if common.h.abs() > (xend - x0).abs() {
            common.h = xend - x0;
        }
        if common.h.abs() < common.hmin {
            common.h = common.hmin * direction;
        }
        common.hold = common.h;
        for i in 0..n {
            work.yh[n + i] = common.h * work.savf[i];
        }

        if let Some(sol) = solout.as_mut() {
            match sol.solout(x0, &mut common.tn, &mut y, None) {
                ControlFlag::Continue | ControlFlag::XOut(_) => {}
                ControlFlag::Interrupt => {
                    return Ok(IntegrationResult::new(
                        common.h,
                        Status::UserInterrupt,
                        Evals {
                            ode: common.nfe,
                            jac: common.nje,
                            lu: common.nlu,
                        },
                        Steps::new(),
                    ));
                }
                ControlFlag::ModifiedSolution => {
                    work.yh[..n].copy_from_slice(&y);
                    system.derivative(common.tn, &y, &mut work.savf);
                    common.nfe += 1;
                    for i in 0..n {
                        work.yh[n + i] = common.h * work.savf[i];
                    }
                }
            }
        }

        let mut status = Status::Success;
        while (xend - common.tn) * direction > 0.0 {
            if common.nst >= self.max_steps {
                status = Status::NeedLargerNMax;
                break;
            }

            let remaining = xend - common.tn;
            if common.h.abs() > remaining.abs() {
                common.h = remaining;
                if common.jstart >= 0 {
                    common.jstart = -2;
                }
            }

            let xold = common.tn;
            stoda(system, &mut y, &rtol, &atol, &mut work, &mut common)?;

            if common.kflag == -1 {
                status = Status::StepSizeTooSmall;
                break;
            }
            if common.kflag == -2 || common.kflag == -3 {
                status = Status::PoorConvergence;
                break;
            }

            y.copy_from_slice(&work.yh[..n]);
            common.total_steps = common.nst + common.rejected;

            let cont = snapshot_step(&common, &work);
            let interpolant = StepInterpolant::new(&cont, xold, common.hu, Self::interpolate);

            if let Some(sol) = solout.as_mut() {
                match sol.solout(xold, &mut common.tn, &mut y, Some(&interpolant)) {
                    ControlFlag::Continue | ControlFlag::XOut(_) => {
                        work.yh[..n].copy_from_slice(&y);
                    }
                    ControlFlag::Interrupt => {
                        status = Status::UserInterrupt;
                        work.yh[..n].copy_from_slice(&y);
                        break;
                    }
                    ControlFlag::ModifiedSolution => {
                        work.yh[..n].copy_from_slice(&y);
                        common.tn = xold + common.hu;
                        restart_after_modification(system, &mut common, &mut work)?;
                    }
                }
            }
        }

        Ok(IntegrationResult::new(
            common.h,
            status,
            Evals {
                ode: common.nfe,
                jac: common.nje,
                lu: common.nlu,
            },
            Steps {
                total: common.total_steps,
                accepted: common.nst,
                rejected: common.rejected,
            },
        ))
    }

    /// Dense output interpolation for stored LSODA snapshots.
    pub fn interpolate(xi: Float, yi: &mut [Float], cont: &[Float], _xold: Float, _h: Float) {
        if cont.len() == yi.len() {
            yi.copy_from_slice(cont);
            return;
        }
        if cont.len() < LSODA_SNAPSHOT_HEADER {
            return;
        }

        let n = cont[0] as usize;
        let nq = cont[1] as usize;
        let tn = cont[2];
        let h = cont[3];
        let _hu = cont[4];
        let _uround = cont[5];
        if yi.len() != n {
            return;
        }
        let l = nq + 1;
        let yh = &cont[LSODA_SNAPSHOT_HEADER..];
        if yh.len() < l * n || h == 0.0 {
            return;
        }

        let s = (xi - tn) / h;
        for i in 0..n {
            yi[i] = yh[i + (l - 1) * n];
        }
        if nq == 0 {
            return;
        }
        for jb in 1..=nq {
            let col = nq - jb;
            for i in 0..n {
                yi[i] = yh[i + col * n] + s * yi[i];
            }
        }
    }
}

fn restart_after_modification<F>(
    system: &F,
    common: &mut Common,
    work: &mut Work,
) -> Result<(), Error>
where
    F: FirstOrderSystem,
{
    system.derivative(common.tn, &work.yh[..common.n], &mut work.savf);
    common.nfe += 1;
    common.meth = MethodFamily::Adams;
    common.miter = 0;
    common.maxord = common.mxordn;
    common.jstart = 0;
    common.mused = 0;
    common.tsw = common.tn;
    common.nq = 1;
    common.l = 2;
    work.yh[common.n..2 * common.n]
        .iter_mut()
        .zip(work.savf.iter())
        .for_each(|(dst, src)| *dst = common.h * *src);
    ewset_scalar_like(&work.yh[..common.n], &mut work.ewt);
    invert_weights(&mut work.ewt)?;
    Ok(())
}

fn validate_common(
    solver: &LSODA,
    n: usize,
    x0: Float,
    xend: Float,
    rtol: &Tolerance,
    atol: &Tolerance,
) -> Result<(), Error> {
    if n == 0 {
        return Ok(());
    }
    if solver.max_steps == 0 {
        return Err(Error::Config(ConfigError::MustBePositive {
            parameter: "max_steps",
            value: solver.max_steps,
        }));
    }
    if solver.maxcor == 0 {
        return Err(Error::Config(ConfigError::MustBePositive {
            parameter: "maxcor",
            value: solver.maxcor,
        }));
    }
    if solver.mxordn == 0 || solver.mxords == 0 {
        return Err(Error::Config(ConfigError::MustBePositive {
            parameter: "mxordn/mxords",
            value: solver.mxordn.min(solver.mxords),
        }));
    }
    if matches!(solver.jac_storage, MatrixStorage::Banded { .. }) {
        return Err(Error::Config(ConfigError::OutOfRange {
            parameter: "jac_storage for LSODA",
            value: 0.0,
            min: 1.0,
            max: 0.0,
        }));
    }
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
    if let Some(h0) = solver.first_step {
        let direction = (xend - x0).signum();
        if h0 == 0.0 || h0.signum() != direction {
            return Err(Error::Config(ConfigError::InvalidStepSize {
                value: h0,
                expected_sign: direction,
            }));
        }
    }
    Ok(())
}

fn ewset(rtol: &Tolerance, atol: &Tolerance, ycur: &[Float], ewt: &mut [Float]) {
    for i in 0..ycur.len() {
        ewt[i] = rtol[i] * ycur[i].abs() + atol[i];
    }
}

fn ewset_scalar_like(ycur: &[Float], ewt: &mut [Float]) {
    for (dst, src) in ewt.iter_mut().zip(ycur.iter()) {
        *dst = src.abs().max(Float::EPSILON);
    }
}

fn invert_weights(ewt: &mut [Float]) -> Result<(), Error> {
    for (index, value) in ewt.iter_mut().enumerate() {
        if *value <= 0.0 {
            return Err(Error::Config(ConfigError::NegativeTolerance {
                kind: "effective",
                index,
                value: *value,
            }));
        }
        *value = 1.0 / *value;
    }
    Ok(())
}

fn vmnorm(v: &[Float], w_inv: &[Float]) -> Float {
    let mut norm: Float = 0.0;
    for i in 0..v.len() {
        norm = norm.max(v[i].abs() * w_inv[i]);
    }
    norm
}

fn fnorm_dense(n: usize, a: &[Float], w_inv: &[Float]) -> Float {
    let mut an: Float = 0.0;
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i + j * n].abs() / w_inv[j];
        }
        an = an.max(sum * w_inv[i]);
    }
    an
}

fn initial_step_size(
    t: Float,
    tout: Float,
    y: &[Float],
    f0: &[Float],
    ewt_inv: &[Float],
    rtol: &Tolerance,
    atol: &Tolerance,
) -> Float {
    let tdist = (tout - t).abs();
    let w0 = t.abs().max(tout.abs());
    let mut tol: Float = 0.0;
    for i in 0..y.len() {
        tol = tol.max(rtol[i]);
    }
    if tol <= 0.0 {
        for i in 0..y.len() {
            if y[i] != 0.0 {
                tol = tol.max(atol[i] / y[i].abs());
            }
        }
    }
    tol = tol.clamp(100.0 * Float::EPSILON, 1.0e-3);
    let mut sum = vmnorm(f0, ewt_inv);
    sum = 1.0 / (tol * w0 * w0) + tol * sum * sum;
    let mut h0 = 1.0 / sum.sqrt();
    h0 = h0.min(tdist);
    h0.copysign(tout - t)
}

fn cfode(
    meth: MethodFamily,
    elco: &mut [Float; ELCO_ROWS * ELCO_COLS],
    tesco: &mut [Float; TESCO_ROWS * ELCO_COLS],
) {
    let mut pc = [0.0; 12];
    match meth {
        MethodFamily::Adams => {
            elco[0] = 1.0;
            elco[1] = 1.0;
            tesco[0] = 0.0;
            tesco[1] = 2.0;
            tesco[3] = 1.0;
            tesco[35] = 0.0;
            pc[0] = 1.0;
            let mut rqfac = 1.0;
            for nq in 1..12 {
                let rq1fac = rqfac;
                rqfac /= nq as Float + 1.0;
                pc[nq] = 0.0;
                for ib in 0..=nq - 1 {
                    pc[nq - ib] = pc[nq - ib - 1] + nq as Float * pc[nq - ib];
                }
                pc[0] *= nq as Float;
                let mut pint = pc[0];
                let mut xpin = 0.5 * pc[0];
                let mut tsign = 1.0;
                for (i, coeff) in pc.iter().enumerate().take(nq + 1).skip(1) {
                    tsign = -tsign;
                    pint += tsign * *coeff / (i as Float + 1.0);
                    xpin += tsign * *coeff / (i as Float + 2.0);
                }
                elco[nq * ELCO_ROWS] = pint * rq1fac;
                elco[1 + nq * ELCO_ROWS] = 1.0;
                for i in 1..=nq {
                    elco[(i + 1) + nq * ELCO_ROWS] = rq1fac * pc[i] / (i as Float + 1.0);
                }
                let agamq = rqfac * xpin;
                let ragq = 1.0 / agamq;
                tesco[1 + TESCO_ROWS * nq] = ragq;
                if nq < 11 {
                    tesco[TESCO_ROWS * (nq + 1)] = ragq * rqfac / (nq as Float + 2.0);
                }
                tesco[2 + TESCO_ROWS * (nq - 1)] = ragq;
            }
        }
        MethodFamily::Bdf => {
            pc[0] = 1.0;
            let mut rq1fac = 1.0;
            for nq in 0..5 {
                pc[nq + 1] = 0.0;
                for ib in 0..=nq {
                    pc[nq - ib + 1] = pc[nq - ib] + (nq as Float + 1.0) * pc[nq - ib + 1];
                }
                pc[0] *= nq as Float + 1.0;
                for i in 0..=nq + 1 {
                    elco[i + nq * ELCO_ROWS] = pc[i] / pc[1];
                }
                elco[1 + nq * ELCO_ROWS] = 1.0;
                tesco[TESCO_ROWS * nq] = rq1fac;
                tesco[1 + TESCO_ROWS * nq] = (nq as Float + 2.0) / elco[nq * ELCO_ROWS];
                tesco[2 + TESCO_ROWS * nq] = (nq as Float + 3.0) / elco[nq * ELCO_ROWS];
                rq1fac /= nq as Float + 1.0;
            }
        }
    }
}

fn stoda_first_call_init(state: &mut Common) {
    state.lmax = state.maxord + 1;
    state.nq = 1;
    state.l = 2;
    state.ialth = 2;
    state.rmax = 10_000.0;
    state.rc = 0.0;
    state.el0 = 1.0;
    state.conv_rate = 0.7;
    state.hold = state.h;
    state.nslp = 0;
    state.ipup = state.miter;
    state.icount = 20;
    state.irflag = 0;
    state.pdest = 0.0;
    state.pdlast = 0.0;
    state.ratio = 5.0;

    cfode(MethodFamily::Bdf, &mut state.elco, &mut state.tesco);
    for i in 0..MXORDS_DEFAULT {
        state.cm2[i] = state.tesco[1 + TESCO_ROWS * i] * state.elco[(i + 1) + ELCO_ROWS * i];
    }
    cfode(MethodFamily::Adams, &mut state.elco, &mut state.tesco);
    for i in 0..ELCO_COLS {
        state.cm1[i] = state.tesco[1 + TESCO_ROWS * i] * state.elco[(i + 1) + ELCO_ROWS * i];
    }
}

fn stoda_reset(state: &mut Common) {
    let nq_idx = state.nq - 1;
    for i in 0..state.l {
        state.el[i] = state.elco[i + nq_idx * ELCO_ROWS];
    }
    state.nqnyh = state.nq * state.n;
    state.rc = state.rc * state.el[0] / state.el0;
    state.el0 = state.el[0];
    state.conit = 0.5 / (state.nq as Float + 2.0);
}

fn stoda_adjust_step_size(state: &mut Common, rh: &mut Float, yh: &mut [Float]) {
    *rh = rh.min(state.rmax);
    *rh /= (1.0_f64 as Float).max(state.h.abs() * state.hmxi * *rh);
    if state.meth != MethodFamily::Bdf {
        state.irflag = 0;
        let pdh = (state.h.abs() * state.pdlast).max(1.0e-6);
        if *rh * pdh * 1.00001 >= SM1[state.nq - 1] {
            *rh = SM1[state.nq - 1] / pdh;
            state.irflag = 1;
        }
    }
    let mut r = 1.0;
    for j in 1..state.l {
        r *= *rh;
        let offset = j * state.n;
        for i in 0..state.n {
            yh[offset + i] *= r;
        }
    }
    state.h *= *rh;
    state.rc *= *rh;
    state.ialth = state.l;
}

fn stoda_get_predicted_values(state: &mut Common, yh: &mut [Float]) {
    if (state.rc - 1.0).abs() > state.ccmax {
        state.ipup = state.miter;
    }
    if state.nst >= state.nslp + state.msbp {
        state.ipup = state.miter;
    }
    state.tn += state.h;
    let mut i1 = state.nqnyh;
    for _ in 0..state.nq {
        i1 -= state.n;
        for i in i1..state.nqnyh {
            yh[i] += yh[i + state.n];
        }
    }
}

fn prja<F>(system: &F, y: &[Float], work: &mut Work, state: &mut Common)
where
    F: FirstOrderSystem,
{
    state.nje += 1;
    state.ierpj = 0;
    state.jcur = 1;
    let hl0 = state.h * state.el0;

    if state.miter == 1 {
        work.jac.fill(0.0);
        system.jac(state.tn, y, &mut work.jac);
        for col in 0..state.n {
            for row in 0..state.n {
                work.wm[2 + row + col * state.n] = -hl0 * work.jac[(row, col)];
            }
        }
    } else {
        let mut fac = vmnorm(&work.savf, &work.ewt);
        let mut r0 = 1000.0 * state.h.abs() * state.uround * state.n as Float * fac;
        if r0 == 0.0 {
            r0 = 1.0;
        }
        let srur = state.uround.sqrt();
        let mut y_fd = y.to_vec();
        for j in 0..state.n {
            let yj = y_fd[j];
            let r = (srur * yj.abs()).max(r0 / work.ewt[j]);
            y_fd[j] = yj + r;
            fac = -hl0 / r;
            system.derivative(state.tn, &y_fd, &mut work.ftem);
            state.nfe += 1;
            for i in 0..state.n {
                work.wm[2 + i + j * state.n] = (work.ftem[i] - work.savf[i]) * fac;
            }
            y_fd[j] = yj;
        }
    }
    state.pdnorm = fnorm_dense(state.n, &work.wm[2..2 + state.n * state.n], &work.ewt) / hl0.abs();
    for i in 0..state.n {
        work.wm[2 + i + i * state.n] += 1.0;
    }
    if dense_lu_factor(
        &mut work.wm[2..2 + state.n * state.n],
        state.n,
        &mut work.ipiv,
    )
    .is_err()
    {
        state.ierpj = 1;
    } else {
        state.nlu += 1;
    }
}

fn solsy(rhs: &mut [Float], work: &Work, state: &mut Common) {
    state.iersl = 0;
    if dense_lu_solve(&work.wm[2..2 + state.n * state.n], state.n, &work.ipiv, rhs).is_err() {
        state.iersl = 1;
    }
}

fn stoda_corrector_loop<F>(
    system: &F,
    y: &mut [Float],
    work: &mut Work,
    state: &mut Common,
) -> CorrectorOutput
where
    F: FirstOrderSystem,
{
    y.copy_from_slice(&work.yh[..state.n]);
    system.derivative(state.tn, y, &mut work.savf);
    state.nfe += 1;

    if state.ipup > 0 {
        prja(system, y, work, state);
        state.ipup = 0;
        state.rc = 1.0;
        state.nslp = state.nst;
        state.conv_rate = 0.7;
        if state.ierpj != 0 {
            return CorrectorOutput {
                status: CorrectorStatus::NoConvergence,
                m: 0,
                del: 0.0,
            };
        }
    }

    work.acor.fill(0.0);

    let pnorm = vmnorm(&work.yh[..state.n], &work.ewt);
    let mut m = 0usize;
    let mut rate: Float = 0.0;
    let mut del: Float = 0.0;
    let mut delp = 0.0;

    loop {
        if state.miter == 0 {
            for i in 0..state.n {
                work.savf[i] = state.h * work.savf[i] - work.yh[state.n + i];
                y[i] = work.savf[i] - work.acor[i];
            }
            del = vmnorm(y, &work.ewt);
            for i in 0..state.n {
                y[i] = work.yh[i] + state.el[0] * work.savf[i];
                work.acor[i] = work.savf[i];
            }
        } else {
            for i in 0..state.n {
                y[i] = state.h * work.savf[i] - (work.yh[state.n + i] + work.acor[i]);
            }
            solsy(y, work, state);
            if state.iersl > 0 {
                if state.jcur != 1 {
                    state.icf = 1;
                    state.ipup = state.miter;
                    return CorrectorOutput {
                        status: CorrectorStatus::Retry,
                        m,
                        del,
                    };
                }
                return CorrectorOutput {
                    status: CorrectorStatus::NoConvergence,
                    m,
                    del,
                };
            }
            del = vmnorm(y, &work.ewt);
            for i in 0..state.n {
                work.acor[i] += y[i];
                y[i] = work.yh[i] + state.el[0] * work.acor[i];
            }
        }

        if del <= 100.0 * pnorm * state.uround {
            break;
        }

        if m != 0 || state.meth != MethodFamily::Adams {
            if m != 0 {
                let mut rm = 1024.0;
                if del <= 1024.0 * delp {
                    rm = del / delp;
                }
                rate = rate.max(rm);
                state.conv_rate = (0.2 * state.conv_rate).max(rm);
            }
            let dcon = del * (1.5 as Float).min(state.conv_rate)
                / (state.tesco[1 + TESCO_ROWS * (state.nq - 1)] * state.conit);
            if dcon <= 1.0 {
                state.pdest = state.pdest.max(rate / (state.h * state.el[0]).abs());
                if state.pdest != 0.0 {
                    state.pdlast = state.pdest;
                }
                break;
            }
        }

        m += 1;
        if m == state.maxcor || (m >= 2 && del > 2.0 * delp) {
            if state.miter != 0 && state.jcur != 1 {
                state.icf = 1;
                state.ipup = state.miter;
                return CorrectorOutput {
                    status: CorrectorStatus::Retry,
                    m,
                    del,
                };
            }
            return CorrectorOutput {
                status: CorrectorStatus::NoConvergence,
                m,
                del,
            };
        }
        delp = del;
        system.derivative(state.tn, y, &mut work.savf);
        state.nfe += 1;
    }

    state.jcur = 0;
    CorrectorOutput {
        status: CorrectorStatus::Converged,
        m,
        del,
    }
}

fn stoda_handle_corrector_failure(
    state: &mut Common,
    work: &mut Work,
    told: Float,
    ncf: &mut usize,
) -> bool {
    state.icf = 2;
    *ncf += 1;
    state.rmax = 2.0;
    state.tn = told;
    retract_yh(state, &mut work.yh);

    if state.h.abs() <= state.hmin * 1.00001 || *ncf == state.mxncf {
        state.kflag = -2;
        return false;
    }

    let mut rh = (0.25 as Float).max(state.hmin / state.h.abs());
    state.ipup = state.miter;
    state.rejected += 1;
    stoda_adjust_step_size(state, &mut rh, &mut work.yh);
    true
}

fn retract_yh(state: &Common, yh: &mut [Float]) {
    let mut i1 = state.nqnyh;
    for _ in 1..=state.nq {
        i1 -= state.n;
        for i in i1..state.nqnyh {
            yh[i] -= yh[i + state.n];
        }
    }
}

fn stoda<F>(
    system: &F,
    y: &mut [Float],
    rtol: &Tolerance,
    atol: &Tolerance,
    work: &mut Work,
    state: &mut Common,
) -> Result<(), Error>
where
    F: FirstOrderSystem,
{
    state.kflag = 0;
    state.ierpj = 0;
    state.iersl = 0;
    state.jcur = 0;
    state.icf = 0;

    match state.jstart {
        0 => {
            stoda_first_call_init(state);
            stoda_reset(state);
        }
        -1 => {
            state.ipup = state.miter;
            state.lmax = state.maxord + 1;
            if state.ialth == 1 {
                state.ialth = 2;
            }
            if state.meth as usize != state.mused {
                cfode(state.meth, &mut state.elco, &mut state.tesco);
                state.ialth = state.l;
                stoda_reset(state);
            }
            if state.h != state.hold {
                let mut rh = (state.h / state.hold).max(state.hmin / state.h.abs());
                state.h = state.hold;
                stoda_adjust_step_size(state, &mut rh, &mut work.yh);
            }
        }
        -2 => {
            if state.h != state.hold {
                let mut rh = state.h / state.hold;
                state.h = state.hold;
                stoda_adjust_step_size(state, &mut rh, &mut work.yh);
            }
        }
        _ => {}
    }

    let told = state.tn;
    stoda_get_predicted_values(state, &mut work.yh);
    let pnorm = vmnorm(&work.yh[..state.n], &work.ewt);
    let mut ncf = 0usize;

    loop {
        let corrector = stoda_corrector_loop(system, y, work, state);
        match corrector.status {
            CorrectorStatus::Retry => continue,
            CorrectorStatus::NoConvergence => {
                if !stoda_handle_corrector_failure(state, work, told, &mut ncf) {
                    state.hold = state.h;
                    state.jstart = 1;
                    return Ok(());
                }
                stoda_get_predicted_values(state, &mut work.yh);
                let _ = pnorm;
                continue;
            }
            CorrectorStatus::Converged => {}
        }

        let dsm = if corrector.m == 0 {
            corrector.del / state.tesco[1 + TESCO_ROWS * (state.nq - 1)]
        } else {
            vmnorm(&work.acor, &work.ewt) / state.tesco[1 + TESCO_ROWS * (state.nq - 1)]
        };
        if dsm > 1.0 {
            state.rejected += 1;
            state.kflag -= 1;
            state.tn = told;
            retract_yh(state, &mut work.yh);
            state.rmax = 2.0;
            if state.h.abs() <= state.hmin * 1.00001 {
                state.kflag = -1;
                state.hold = state.h;
                state.jstart = 1;
                return Ok(());
            }
            if state.kflag <= -3 {
                if state.kflag == -10 {
                    state.kflag = -1;
                    state.hold = state.h;
                    state.jstart = 1;
                    return Ok(());
                }
                let rh = (0.1 as Float).max(state.hmin / state.h.abs());
                state.h *= rh;
                y.copy_from_slice(&work.yh[..state.n]);
                system.derivative(state.tn, y, &mut work.savf);
                state.nfe += 1;
                for i in 0..state.n {
                    work.yh[state.n + i] = state.h * work.savf[i];
                }
                state.ipup = state.miter;
                state.ialth = 5;
                if state.nq != 1 {
                    state.nq = 1;
                    state.l = 2;
                    stoda_reset(state);
                }
                stoda_get_predicted_values(state, &mut work.yh);
                let _ = vmnorm(&work.yh[..state.n], &work.ewt);
                continue;
            }
            let mut rh = (1.0 / (1.2 * dsm.powf(1.0 / state.l as Float) + 1.2e-6)).min(1.0);
            if state.kflag <= -2 {
                rh = rh.min(0.2);
            }
            rh = rh.max(state.hmin / state.h.abs());
            stoda_adjust_step_size(state, &mut rh, &mut work.yh);
            stoda_get_predicted_values(state, &mut work.yh);
            continue;
        }

        state.kflag = 0;
        state.nst += 1;
        state.hu = state.h;
        state.nqu = state.nq;
        state.mused = state.meth as usize;
        for j in 0..state.l {
            for i in 0..state.n {
                work.yh[i + j * state.n] += state.el[j] * work.acor[i];
            }
        }

        state.icount -= 1;
        if state.icount < 0 {
            if state.meth == MethodFamily::Adams {
                if state.nq <= 5 && (dsm > 100.0 * pnorm * state.uround) && (state.pdest != 0.0) {
                    let exsm = 1.0 / state.l as Float;
                    let mut rh1 = 1.0 / (1.2 * dsm.powf(exsm) + 1.2e-6);
                    let mut rh1it = 2.0 * rh1;
                    let pdh = state.pdlast * state.h.abs();
                    if pdh * rh1 > 1.0e-5 {
                        rh1it = SM1[state.nq - 1] / pdh;
                    }
                    rh1 = rh1.min(rh1it);
                    let (rh2, nqm2) = if state.nq > state.mxords {
                        let lm2 = state.mxords + 1;
                        let dm2 = vmnorm(&work.yh[lm2 * state.n..(lm2 + 1) * state.n], &work.ewt)
                            / state.cm2[state.mxords - 1];
                        (
                            1.0 / (1.2 * dm2.powf(1.0 / lm2 as Float) + 1.2e-6),
                            state.mxords,
                        )
                    } else {
                        let dm2 = dsm * (state.cm1[state.nq - 1] / state.cm2[state.nq - 1]);
                        (1.0 / (1.2 * dm2.powf(exsm) + 1.2e-6), state.nq)
                    };
                    if rh2 >= state.ratio * rh1 {
                        let mut rh = rh2.max(state.hmin / state.h.abs());
                        state.icount = 20;
                        state.meth = MethodFamily::Bdf;
                        state.miter = state.jtyp;
                        state.pdlast = 0.0;
                        state.nq = nqm2;
                        state.l = state.nq + 1;
                        state.maxord = state.mxords;
                        stoda_adjust_step_size(state, &mut rh, &mut work.yh);
                        break;
                    }
                }
            } else {
                let exsm = 1.0 / state.l as Float;
                let (dm1, rh1, nqm1, exm1) = if state.mxordn >= state.nq {
                    let dm1 = dsm * (state.cm2[state.nq - 1] / state.cm1[state.nq - 1]);
                    let rh1 = 1.0 / (1.2 * dm1.powf(exsm) + 1.2e-6);
                    (dm1, rh1, state.nq, exsm)
                } else {
                    let lm1 = state.mxordn + 1;
                    let dm1 = vmnorm(&work.yh[lm1 * state.n..(lm1 + 1) * state.n], &work.ewt)
                        / state.cm1[state.mxordn - 1];
                    (
                        dm1,
                        1.0 / (1.2 * dm1.powf(1.0 / lm1 as Float) + 1.2e-6),
                        state.mxordn,
                        1.0 / lm1 as Float,
                    )
                };
                let mut rh1it = 2.0 * rh1;
                let pdh = state.pdnorm * state.h.abs();
                if pdh * rh1 > 1.0e-5 {
                    rh1it = SM1[nqm1 - 1] / pdh;
                }
                let rh1 = rh1.min(rh1it);
                let rh2 = 1.0 / (1.2 * dsm.powf(exsm) + 1.2e-6);
                if rh1 * state.ratio >= 5.0 * rh2 {
                    let dm1 = (rh1.max(0.001)).powf(exm1) * dm1;
                    if dm1 > 1000.0 * state.uround * pnorm {
                        let mut rh = rh1.max(state.hmin / state.h.abs());
                        state.icount = 20;
                        state.meth = MethodFamily::Adams;
                        state.miter = 0;
                        state.pdlast = 0.0;
                        state.nq = nqm1;
                        state.l = state.nq + 1;
                        state.maxord = state.mxordn;
                        stoda_adjust_step_size(state, &mut rh, &mut work.yh);
                        break;
                    }
                }
            }
        }

        state.ialth -= 1;
        if state.ialth == 0 {
            let exsm = 1.0 / state.l as Float;
            let mut rhup = 0.0;
            if state.l != state.lmax {
                for i in 0..state.n {
                    work.savf[i] = work.acor[i] - work.yh[i + (state.lmax - 1) * state.n];
                }
                let dup =
                    vmnorm(&work.savf, &work.ewt) / state.tesco[2 + TESCO_ROWS * (state.nq - 1)];
                rhup = 1.0 / (1.4 * dup.powf(1.0 / (state.l as Float + 1.0)) + 1.4e-6);
            }
            let mut rhsm = 1.0 / (1.2 * dsm.powf(exsm) + 1.2e-6);
            let mut rhdn = 0.0;
            if state.nq != 1 {
                let ddn = vmnorm(
                    &work.yh[(state.l - 1) * state.n..state.l * state.n],
                    &work.ewt,
                ) / state.tesco[TESCO_ROWS * (state.nq - 1)];
                rhdn = 1.0 / (1.3 * ddn.powf(1.0 / state.nq as Float) + 1.3e-6);
            }
            if state.meth == MethodFamily::Adams {
                let pdh = (state.h.abs() * state.pdlast).max(1.0e-6);
                if state.l < state.lmax {
                    rhup = rhup.min(SM1[state.l - 1] / pdh);
                }
                rhsm = rhsm.min(SM1[state.nq - 1] / pdh);
                if state.nq > 1 {
                    rhdn = rhdn.min(SM1[state.nq - 2] / pdh);
                }
                state.pdest = 0.0;
            }

            let (newq, mut rh) = if rhsm >= rhup && rhsm >= rhdn {
                (state.nq, rhsm)
            } else if rhup > rhdn {
                let newq = state.l;
                if rhup < 1.1 {
                    state.ialth = 3;
                    break;
                }
                let r = state.el[state.l - 1] / state.l as Float;
                for i in 0..state.n {
                    work.yh[i + newq * state.n] = work.acor[i] * r;
                }
                (newq, rhup)
            } else {
                (state.nq - 1, rhdn)
            };

            if state.meth == MethodFamily::Adams
                && rh * (state.h.abs() * state.pdlast).max(1.0e-6) * 1.00001 >= SM1[newq - 1]
            {
                state.ialth = 3;
                break;
            }
            if state.kflag == 0 && rh < 1.1 {
                state.ialth = 3;
                break;
            }
            if state.kflag <= -2 {
                rh = rh.min(0.2);
            }
            if newq != state.nq {
                state.nq = newq;
                state.l = state.nq + 1;
                stoda_reset(state);
            }
            rh = rh.max(state.hmin / state.h.abs());
            stoda_adjust_step_size(state, &mut rh, &mut work.yh);
        }
        break;
    }

    let r = 1.0 / state.tesco[1 + TESCO_ROWS * (state.nqu.saturating_sub(1))];
    for value in &mut work.acor {
        *value *= r;
    }
    state.hold = state.h;
    state.jstart = 1;
    ewset(rtol, atol, &work.yh[..state.n], &mut work.ewt);
    invert_weights(&mut work.ewt)?;
    Ok(())
}

fn snapshot_step(state: &Common, work: &Work) -> Vec<Float> {
    let l = state.nq + 1;
    let mut cont = Vec::with_capacity(LSODA_SNAPSHOT_HEADER + l * state.n);
    cont.push(state.n as Float);
    cont.push(state.nq as Float);
    cont.push(state.tn);
    cont.push(state.h);
    cont.push(state.hu);
    cont.push(state.uround);
    cont.extend_from_slice(&work.yh[..l * state.n]);
    cont
}

fn dense_lu_factor(a: &mut [Float], n: usize, ipiv: &mut [usize]) -> Result<(), ()> {
    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = a[k + k * n].abs();
        for i in k + 1..n {
            let val = a[i + k * n].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = i;
            }
        }
        if pivot_val == 0.0 {
            return Err(());
        }
        ipiv[k] = pivot_row;
        if pivot_row != k {
            for j in 0..n {
                a.swap(k + j * n, pivot_row + j * n);
            }
        }
        let diag = a[k + k * n];
        for i in k + 1..n {
            a[i + k * n] /= diag;
        }
        for j in k + 1..n {
            let ukj = a[k + j * n];
            for i in k + 1..n {
                a[i + j * n] -= a[i + k * n] * ukj;
            }
        }
    }
    Ok(())
}

fn dense_lu_solve(a: &[Float], n: usize, ipiv: &[usize], b: &mut [Float]) -> Result<(), ()> {
    for k in 0..n {
        let p = ipiv[k];
        if p != k {
            b.swap(k, p);
        }
        for i in k + 1..n {
            b[i] -= a[i + k * n] * b[k];
        }
    }
    for kb in 0..n {
        let k = n - 1 - kb;
        let diag = a[k + k * n];
        if diag == 0.0 {
            return Err(());
        }
        b[k] /= diag;
        for i in 0..k {
            b[i] -= a[i + k * n] * b[k];
        }
    }
    Ok(())
}
