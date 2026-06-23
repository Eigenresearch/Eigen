use pyo3::prelude::*;

// Direct Rust kernels for statevector updates

#[pyfunction]
pub fn apply_h(mut state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let step = 1 << k;
    let inv_sqrt2 = 0.7071067811865475;
    for i in (0..n).step_by(step * 2) {
        for j in i..(i + step) {
            let i0 = j;
            let i1 = j + step;
            let (a0_re, a0_im) = state[i0];
            let (a1_re, a1_im) = state[i1];
            state[i0] = ((a0_re + a1_re) * inv_sqrt2, (a0_im + a1_im) * inv_sqrt2);
            state[i1] = ((a0_re - a1_re) * inv_sqrt2, (a0_im - a1_im) * inv_sqrt2);
        }
    }
    Ok(state)
}

#[pyfunction]
pub fn apply_x(mut state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let step = 1 << k;
    for i in (0..n).step_by(step * 2) {
        for j in i..(i + step) {
            let i0 = j;
            let i1 = j + step;
            state.swap(i0, i1);
        }
    }
    Ok(state)
}

#[pyfunction]
pub fn apply_y(mut state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let step = 1 << k;
    for i in (0..n).step_by(step * 2) {
        for j in i..(i + step) {
            let i0 = j;
            let i1 = j + step;
            let (a0_re, a0_im) = state[i0];
            let (a1_re, a1_im) = state[i1];
            // a0' = -i * a1 = (a1_im, -a1_re)
            // a1' = i * a0 = (-a0_im, a0_re)
            state[i0] = (a1_im, -a1_re);
            state[i1] = (-a0_im, a0_re);
        }
    }
    Ok(state)
}

#[pyfunction]
pub fn apply_z(mut state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let step = 1 << k;
    for i in (0..n).step_by(step * 2) {
        for j in i..(i + step) {
            let i1 = j + step;
            let (re, im) = state[i1];
            state[i1] = (-re, -im);
        }
    }
    Ok(state)
}

#[pyfunction]
pub fn apply_s(mut state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let step = 1 << k;
    for i in (0..n).step_by(step * 2) {
        for j in i..(i + step) {
            let i1 = j + step;
            let (re, im) = state[i1];
            // i * (re + i*im) = -im + i*re
            state[i1] = (-im, re);
        }
    }
    Ok(state)
}

#[pyfunction]
pub fn apply_t(mut state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let step = 1 << k;
    let cos_t = 0.7071067811865475;
    let sin_t = 0.7071067811865475;
    for i in (0..n).step_by(step * 2) {
        for j in i..(i + step) {
            let i1 = j + step;
            let (re, im) = state[i1];
            // (cos + i*sin) * (re + i*im) = (re*cos - im*sin) + i*(re*sin + im*cos)
            state[i1] = (re * cos_t - im * sin_t, re * sin_t + im * cos_t);
        }
    }
    Ok(state)
}

#[pyfunction]
pub fn apply_cnot(mut state: Vec<(f64, f64)>, control: usize, target: usize) -> PyResult<Vec<(f64, f64)>> {
    let n = state.len();
    let c_mask = 1 << control;
    let t_mask = 1 << target;
    for i in 0..n {
        if (i & c_mask) != 0 && (i & t_mask) == 0 {
            let i_target_1 = i | t_mask;
            state.swap(i, i_target_1);
        }
    }
    Ok(state)
}
