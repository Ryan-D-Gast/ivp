use ivp::prelude::*;

pub struct SHO;
impl FirstOrderSystem for SHO {
    fn derivative(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
}

#[allow(dead_code)]
pub fn default_ivp_dense<'a, F>(
    system: &'a F,
    x0: f64,
    xend: f64,
    y0: &'a [f64],
    method: Method,
) -> ivp::solve::FirstOrderIvp<'a, F>
where
    F: FirstOrderSystem,
{
    Ivp::first_order(system, x0, xend, y0)
        .method(method)
        .rtol(1e-9)
        .atol(1e-9)
        .dense_output(true)
}

#[allow(dead_code)]
pub fn default_ivp<'a, F>(
    system: &'a F,
    x0: f64,
    xend: f64,
    y0: &'a [f64],
    method: Method,
) -> ivp::solve::FirstOrderIvp<'a, F>
where
    F: FirstOrderSystem,
{
    Ivp::first_order(system, x0, xend, y0)
        .method(method)
        .rtol(1e-9)
        .atol(1e-9)
}
