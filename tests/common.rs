use ivp::prelude::*;

pub struct SHO;
impl IVP for SHO {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
}

#[allow(dead_code)]
pub fn default_opts_dense(method: Method) -> Options {
    Options::builder()
        .method(method)
        .rtol(1e-9)
        .atol(1e-9)
        .dense_output(true)
        .build()
}

#[allow(dead_code)]
pub fn default_opts(method: Method) -> Options {
    Options::builder()
        .method(method)
        .rtol(1e-9)
        .atol(1e-9)
        .build()
}
