#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
use std::f64::consts::FRAC_1_SQRT_2;
use std::collections::HashMap;

// --- Free Functions delegating to optimized RustStatevector ---

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_h(state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_h(k).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_x(state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_x(k).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_y(state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_y(k).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_z(state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_z(k).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_s(state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_s(k).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_t(state: Vec<(f64, f64)>, k: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_t(k).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn apply_cnot(state: Vec<(f64, f64)>, control: usize, target: usize) -> PyResult<Vec<(f64, f64)>> {
    let num_qubits = (state.len() as f64).log2().round() as usize;
    let mut sv = RustStatevector { state, num_qubits };
    sv.apply_cnot(control, target).map_err(|e| PyValueError::new_err(e))?;
    Ok(sv.state)
}

// --- Stateful RustStatevector PyClass for Zero-Copy Backend ---

#[cfg_attr(feature = "pyo3", pyclass)]
pub struct RustStatevector {
    state: Vec<(f64, f64)>,
    num_qubits: usize,
}

impl RustStatevector {
    pub fn new() -> Self {
        RustStatevector {
            state: vec![(1.0, 0.0)],
            num_qubits: 0,
        }
    }

    pub fn allocate_qubit(&mut self) -> Result<(), String> {
        let n = self.state.len();
        if n >= (1 << 25) {
            return Err("Dense simulation is limited to 25 qubits to prevent memory exhaustion.".to_string());
        }
        self.state.resize(n * 2, (0.0, 0.0));
        self.num_qubits += 1;
        Ok(())
    }

    pub fn get_state(&self) -> Vec<(f64, f64)> {
        self.state.clone()
    }

    pub fn set_state(&mut self, new_state: Vec<(f64, f64)>) -> Result<(), String> {
        let n = new_state.len();
        if n != (1 << self.num_qubits) {
            return Err(format!(
                "Invalid state length: expected {}, got {}",
                1 << self.num_qubits,
                n
            ));
        }
        self.state = new_state;
        Ok(())
    }

    // High performance gate dispatcher loop supporting rayon multithreading for N >= 16384
    fn run_gate_loop<F>(&mut self, step: usize, f: F)
    where
        F: Fn((f64, f64), (f64, f64)) -> ((f64, f64), (f64, f64)) + Sync + Send,
    {
        let n = self.state.len();
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            let chunks = n / (step * 2);
            (0..chunks).into_par_iter().for_each(|c| {
                let i = c * step * 2;
                unsafe {
                    let ptr = state_ptr as *mut (f64, f64);
                    for j in i..(i + step) {
                        let i0 = j;
                        let i1 = j + step;
                        let a0 = *ptr.add(i0);
                        let a1 = *ptr.add(i1);
                        let (r0, r1) = f(a0, a1);
                        *ptr.add(i0) = r0;
                        *ptr.add(i1) = r1;
                    }
                }
            });
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let i0 = j;
                    let i1 = j + step;
                    let a0 = self.state[i0];
                    let a1 = self.state[i1];
                    let (r0, r1) = f(a0, a1);
                    self.state[i0] = r0;
                    self.state[i1] = r1;
                }
            }
        }
    }

    pub fn apply_h(&mut self, k: usize) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        let inv_sqrt2 = FRAC_1_SQRT_2;
        self.run_gate_loop(step, |(a0_re, a0_im), (a1_re, a1_im)| {
            (
                ((a0_re + a1_re) * inv_sqrt2, (a0_im + a1_im) * inv_sqrt2),
                ((a0_re - a1_re) * inv_sqrt2, (a0_im - a1_im) * inv_sqrt2),
            )
        });
        Ok(())
    }

    pub fn apply_x(&mut self, k: usize) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        self.run_gate_loop(step, |a0, a1| (a1, a0));
        Ok(())
    }

    pub fn apply_y(&mut self, k: usize) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        self.run_gate_loop(step, |(a0_re, a0_im), (a1_re, a1_im)| {
            ((a1_im, -a1_re), (-a0_im, a0_re))
        });
        Ok(())
    }

    pub fn apply_z(&mut self, k: usize) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        self.run_gate_loop(step, |a0, (a1_re, a1_im)| {
            (a0, (-a1_re, -a1_im))
        });
        Ok(())
    }

    pub fn apply_s(&mut self, k: usize) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        self.run_gate_loop(step, |a0, (a1_re, a1_im)| {
            (a0, (-a1_im, a1_re))
        });
        Ok(())
    }

    pub fn apply_t(&mut self, k: usize) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        let cos_t = FRAC_1_SQRT_2;
        let sin_t = FRAC_1_SQRT_2;
        self.run_gate_loop(step, |a0, (a1_re, a1_im)| {
            (a0, (a1_re * cos_t - a1_im * sin_t, a1_re * sin_t + a1_im * cos_t))
        });
        Ok(())
    }

    pub fn apply_rx(&mut self, k: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        self.run_gate_loop(step, |(a0_re, a0_im), (a1_re, a1_im)| {
            (
                (cos_val * a0_re + sin_val * a1_im, cos_val * a0_im - sin_val * a1_re),
                (sin_val * a0_im + cos_val * a1_re, -sin_val * a0_re + cos_val * a1_im),
            )
        });
        Ok(())
    }

    pub fn apply_ry(&mut self, k: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        self.run_gate_loop(step, |(a0_re, a0_im), (a1_re, a1_im)| {
            (
                (cos_val * a0_re - sin_val * a1_re, cos_val * a0_im - sin_val * a1_im),
                (sin_val * a0_re + cos_val * a1_re, sin_val * a0_im + cos_val * a1_im),
            )
        });
        Ok(())
    }

    pub fn apply_rz(&mut self, k: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        self.run_gate_loop(step, |(a0_re, a0_im), (a1_re, a1_im)| {
            (
                (a0_re * cos_val + a0_im * sin_val, a0_im * cos_val - a0_re * sin_val),
                (a1_re * cos_val - a1_im * sin_val, a1_im * cos_val + a1_re * sin_val),
            )
        });
        Ok(())
    }

    pub fn apply_1qubit_gate(
        &mut self,
        k: usize,
        u00_re: f64, u00_im: f64,
        u01_re: f64, u01_im: f64,
        u10_re: f64, u10_im: f64,
        u11_re: f64, u11_im: f64,
    ) -> Result<(), String> {
        let n = self.state.len();
        let step = 1 << k;
        if step >= n {
            return Err(format!("Qubit index out of bounds: k={}, state size={}", k, n));
        }
        self.run_gate_loop(step, |(a0_re, a0_im), (a1_re, a1_im)| {
            let r0_re = (u00_re * a0_re - u00_im * a0_im) + (u01_re * a1_re - u01_im * a1_im);
            let r0_im = (u00_re * a0_im + u00_im * a0_re) + (u01_re * a1_im + u01_im * a1_re);
            let r1_re = (u10_re * a0_re - u10_im * a0_im) + (u11_re * a1_re - u11_im * a1_im);
            let r1_im = (u10_re * a0_im + u10_im * a0_re) + (u11_re * a1_im + u11_im * a1_re);
            ((r0_re, r0_im), (r1_re, r1_im))
        });
        Ok(())
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        if c_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, target={}, state size={}",
                control, target, n
            ));
        }
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        std::ptr::swap(ptr.add(i), ptr.add(i_target_1));
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    self.state.swap(i, i_target_1);
                }
            }
        }
        Ok(())
    }

    pub fn apply_cz(&mut self, control: usize, target: usize) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        if c_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, target={}, state size={}",
                control, target, n
            ));
        }
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 && (i & t_mask) != 0 {
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        let (re, im) = *ptr.add(i);
                        *ptr.add(i) = (-re, -im);
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 && (i & t_mask) != 0 {
                    let (re, im) = self.state[i];
                    self.state[i] = (-re, -im);
                }
            }
        }
        Ok(())
    }

    pub fn apply_swap(&mut self, q1: usize, q2: usize) -> Result<(), String> {
        let n = self.state.len();
        let mask1 = 1 << q1;
        let mask2 = 1 << q2;
        if mask1 >= n || mask2 >= n {
            return Err(format!(
                "Qubit index out of bounds: q1={}, q2={}, state size={}",
                q1, q2, n
            ));
        }
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & mask1) != 0 && (i & mask2) == 0 {
                    let j = (i & !mask1) | mask2;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        std::ptr::swap(ptr.add(i), ptr.add(j));
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & mask1) != 0 && (i & mask2) == 0 {
                    let j = (i & !mask1) | mask2;
                    self.state.swap(i, j);
                }
            }
        }
        Ok(())
    }

    pub fn apply_ccx(&mut self, control1: usize, control2: usize, target: usize) -> Result<(), String> {
        let n = self.state.len();
        let c1_mask = 1 << control1;
        let c2_mask = 1 << control2;
        let t_mask = 1 << target;
        if c1_mask >= n || c2_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control1={}, control2={}, target={}, state size={}",
                control1, control2, target, n
            ));
        }
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c1_mask) != 0 && (i & c2_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        std::ptr::swap(ptr.add(i), ptr.add(i_target_1));
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c1_mask) != 0 && (i & c2_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    self.state.swap(i, i_target_1);
                }
            }
        }
        Ok(())
    }

    pub fn apply_cswap(&mut self, control: usize, q1: usize, q2: usize) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let mask1 = 1 << q1;
        let mask2 = 1 << q2;
        if c_mask >= n || mask1 >= n || mask2 >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, q1={}, q2={}, state size={}",
                control, q1, q2, n
            ));
        }
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 && (i & mask1) != 0 && (i & mask2) == 0 {
                    let j = (i & !mask1) | mask2;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        std::ptr::swap(ptr.add(i), ptr.add(j));
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 && (i & mask1) != 0 && (i & mask2) == 0 {
                    let j = (i & !mask1) | mask2;
                    self.state.swap(i, j);
                }
            }
        }
        Ok(())
    }

    pub fn apply_cp(&mut self, control: usize, target: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        if c_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, target={}, state size={}",
                control, target, n
            ));
        }
        let cos_val = theta.cos();
        let sin_val = theta.sin();
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 && (i & t_mask) != 0 {
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        let (re, im) = *ptr.add(i);
                        *ptr.add(i) = (re * cos_val - im * sin_val, re * sin_val + im * cos_val);
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 && (i & t_mask) != 0 {
                    let (re, im) = self.state[i];
                    self.state[i] = (re * cos_val - im * sin_val, re * sin_val + im * cos_val);
                }
            }
        }
        Ok(())
    }

    pub fn apply_crx(&mut self, control: usize, target: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        if c_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, target={}, state size={}",
                control, target, n
            ));
        }
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        let (a0_re, a0_im) = *ptr.add(i);
                        let (a1_re, a1_im) = *ptr.add(i_target_1);
                        *ptr.add(i) = (cos_val * a0_re + sin_val * a1_im, cos_val * a0_im - sin_val * a1_re);
                        *ptr.add(i_target_1) = (sin_val * a0_im + cos_val * a1_re, -sin_val * a0_re + cos_val * a1_im);
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    let (a0_re, a0_im) = self.state[i];
                    let (a1_re, a1_im) = self.state[i_target_1];
                    self.state[i] = (cos_val * a0_re + sin_val * a1_im, cos_val * a0_im - sin_val * a1_re);
                    self.state[i_target_1] = (sin_val * a0_im + cos_val * a1_re, -sin_val * a0_re + cos_val * a1_im);
                }
            }
        }
        Ok(())
    }

    pub fn apply_cry(&mut self, control: usize, target: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        if c_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, target={}, state size={}",
                control, target, n
            ));
        }
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        let (a0_re, a0_im) = *ptr.add(i);
                        let (a1_re, a1_im) = *ptr.add(i_target_1);
                        *ptr.add(i) = (cos_val * a0_re - sin_val * a1_re, cos_val * a0_im - sin_val * a1_im);
                        *ptr.add(i_target_1) = (sin_val * a0_re + cos_val * a1_re, sin_val * a0_im + cos_val * a1_im);
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 && (i & t_mask) == 0 {
                    let i_target_1 = i | t_mask;
                    let (a0_re, a0_im) = self.state[i];
                    let (a1_re, a1_im) = self.state[i_target_1];
                    self.state[i] = (cos_val * a0_re - sin_val * a1_re, cos_val * a0_im - sin_val * a1_im);
                    self.state[i_target_1] = (sin_val * a0_re + cos_val * a1_re, sin_val * a0_im + cos_val * a1_im);
                }
            }
        }
        Ok(())
    }

    pub fn apply_crz(&mut self, control: usize, target: usize, theta: f64) -> Result<(), String> {
        let n = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        if c_mask >= n || t_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: control={}, target={}, state size={}",
                control, target, n
            ));
        }
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        use rayon::prelude::*;
        if n >= 16384 {
            let state_ptr = self.state.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                if (i & c_mask) != 0 {
                    let has_target = (i & t_mask) != 0;
                    unsafe {
                        let ptr = state_ptr as *mut (f64, f64);
                        let (re, im) = *ptr.add(i);
                        if !has_target {
                            *ptr.add(i) = (re * cos_val + im * sin_val, im * cos_val - re * sin_val);
                        } else {
                            *ptr.add(i) = (re * cos_val - im * sin_val, im * cos_val + re * sin_val);
                        }
                    }
                }
            });
        } else {
            for i in 0..n {
                if (i & c_mask) != 0 {
                    let has_target = (i & t_mask) != 0;
                    let (re, im) = self.state[i];
                    if !has_target {
                        self.state[i] = (re * cos_val + im * sin_val, im * cos_val - re * sin_val);
                    } else {
                        self.state[i] = (re * cos_val - im * sin_val, im * cos_val + re * sin_val);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn measure(&mut self, k: usize, r: f64) -> Result<usize, String> {
        let n = self.state.len();
        let k_mask = 1 << k;
        if k_mask >= n {
            return Err(format!(
                "Qubit index out of bounds: k={}, state size={}",
                k, n
            ));
        }
        
        let mut p0 = 0.0;
        for (i, &(re, im)) in self.state.iter().enumerate() {
            if (i & k_mask) == 0 {
                p0 += re * re + im * im;
            }
        }
        
        let outcome = if r < p0 {
            let norm = p0.sqrt();
            let norm_val = if norm > 1e-15 { norm } else { 1.0 };
            for i in 0..n {
                if (i & k_mask) != 0 {
                    self.state[i] = (0.0, 0.0);
                } else {
                    let (re, im) = self.state[i];
                    self.state[i] = (re / norm_val, im / norm_val);
                }
            }
            0
        } else {
            let p1 = 1.0 - p0;
            let norm = p1.sqrt();
            let norm_val = if norm > 1e-15 { norm } else { 1.0 };
            for i in 0..n {
                if (i & k_mask) == 0 {
                    self.state[i] = (0.0, 0.0);
                } else {
                    let (re, im) = self.state[i];
                    self.state[i] = (re / norm_val, im / norm_val);
                }
            }
            1
        };
        Ok(outcome)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RustStatevector {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[pyo3(name = "allocate_qubit")]
    pub fn allocate_qubit_py(&mut self) -> PyResult<()> {
        self.allocate_qubit().map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    #[pyo3(name = "get_state")]
    pub fn get_state_py(&self) -> Vec<(f64, f64)> {
        self.get_state()
    }

    #[pyo3(name = "set_state")]
    pub fn set_state_py(&mut self, new_state: Vec<(f64, f64)>) -> PyResult<()> {
        self.set_state(new_state).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_h")]
    pub fn apply_h_py(&mut self, k: usize) -> PyResult<()> {
        self.apply_h(k).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_x")]
    pub fn apply_x_py(&mut self, k: usize) -> PyResult<()> {
        self.apply_x(k).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_y")]
    pub fn apply_y_py(&mut self, k: usize) -> PyResult<()> {
        self.apply_y(k).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_z")]
    pub fn apply_z_py(&mut self, k: usize) -> PyResult<()> {
        self.apply_z(k).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_s")]
    pub fn apply_s_py(&mut self, k: usize) -> PyResult<()> {
        self.apply_s(k).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_t")]
    pub fn apply_t_py(&mut self, k: usize) -> PyResult<()> {
        self.apply_t(k).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_rx")]
    pub fn apply_rx_py(&mut self, k: usize, theta: f64) -> PyResult<()> {
        self.apply_rx(k, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_ry")]
    pub fn apply_ry_py(&mut self, k: usize, theta: f64) -> PyResult<()> {
        self.apply_ry(k, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_rz")]
    pub fn apply_rz_py(&mut self, k: usize, theta: f64) -> PyResult<()> {
        self.apply_rz(k, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_1qubit_gate")]
    pub fn apply_1qubit_gate_py(
        &mut self,
        k: usize,
        u00_re: f64, u00_im: f64,
        u01_re: f64, u01_im: f64,
        u10_re: f64, u10_im: f64,
        u11_re: f64, u11_im: f64,
    ) -> PyResult<()> {
        self.apply_1qubit_gate(k, u00_re, u00_im, u01_re, u01_im, u10_re, u10_im, u11_re, u11_im)
            .map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_cnot")]
    pub fn apply_cnot_py(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_cnot(control, target).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_cz")]
    pub fn apply_cz_py(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_cz(control, target).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_swap")]
    pub fn apply_swap_py(&mut self, q1: usize, q2: usize) -> PyResult<()> {
        self.apply_swap(q1, q2).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_ccx")]
    pub fn apply_ccx_py(&mut self, control1: usize, control2: usize, target: usize) -> PyResult<()> {
        self.apply_ccx(control1, control2, target).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_cswap")]
    pub fn apply_cswap_py(&mut self, control: usize, q1: usize, q2: usize) -> PyResult<()> {
        self.apply_cswap(control, q1, q2).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_cp")]
    pub fn apply_cp_py(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_cp(control, target, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_crx")]
    pub fn apply_crx_py(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_crx(control, target, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_cry")]
    pub fn apply_cry_py(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_cry(control, target, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "apply_crz")]
    pub fn apply_crz_py(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_crz(control, target, theta).map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(name = "measure")]
    pub fn measure_py(&mut self, k: usize, r: f64) -> PyResult<usize> {
        self.measure(k, r).map_err(|e| PyValueError::new_err(e))
    }
}

#[cfg_attr(feature = "pyo3", pyclass)]
pub struct RustSparseSimulator {
    state: HashMap<u64, (f64, f64)>,
    num_qubits: usize,
}

impl RustSparseSimulator {
    pub fn new() -> Self {
        let mut state = HashMap::new();
        state.insert(0, (1.0, 0.0));
        RustSparseSimulator { state, num_qubits: 0 }
    }

    pub fn allocate_qubit(&mut self) {
        self.num_qubits += 1;
    }

    pub fn apply_1qubit_gate(
        &mut self,
        k: usize,
        u00_re: f64, u00_im: f64,
        u01_re: f64, u01_im: f64,
        u10_re: f64, u10_im: f64,
        u11_re: f64, u11_im: f64,
    ) {
        let mask = 1 << k;
        let mut groups: HashMap<u64, [(f64, f64); 2]> = HashMap::new();
        
        for (&key, &amp) in &self.state {
            let base = key & !mask;
            let bit = (key & mask) != 0;
            let entry = groups.entry(base).or_insert([(0.0, 0.0), (0.0, 0.0)]);
            if bit {
                entry[1] = amp;
            } else {
                entry[0] = amp;
            }
        }
        
        let mut new_state = HashMap::new();
        let u00 = (u00_re, u00_im);
        let u01 = (u01_re, u01_im);
        let u10 = (u10_re, u10_im);
        let u11 = (u11_re, u11_im);
        
        for (base, [a0, a1]) in groups {
            let v0 = (
                (u00.0 * a0.0 - u00.1 * a0.1) + (u01.0 * a1.0 - u01.1 * a1.1),
                (u00.0 * a0.1 + u00.1 * a0.0) + (u01.0 * a1.1 + u01.1 * a1.0),
            );
            let v1 = (
                (u10.0 * a0.0 - u10.1 * a0.1) + (u11.0 * a1.0 - u11.1 * a1.1),
                (u10.0 * a0.1 + u10.1 * a0.0) + (u11.0 * a1.1 + u11.1 * a1.0),
            );
            
            if v0.0.powi(2) + v0.1.powi(2) > 1e-24 {
                new_state.insert(base, v0);
            }
            if v1.0.powi(2) + v1.1.powi(2) > 1e-24 {
                new_state.insert(base | mask, v1);
            }
        }
        self.state = new_state;
    }

    pub fn apply_h(&mut self, k: usize) {
        let inv_sqrt2 = FRAC_1_SQRT_2;
        self.apply_1qubit_gate(k, inv_sqrt2, 0.0, inv_sqrt2, 0.0, inv_sqrt2, 0.0, -inv_sqrt2, 0.0);
    }

    pub fn apply_x(&mut self, k: usize) {
        let mask = 1 << k;
        let mut new_state = HashMap::new();
        for (key, amp) in &self.state {
            new_state.insert(key ^ mask, *amp);
        }
        self.state = new_state;
    }

    pub fn apply_y(&mut self, k: usize) {
        let mask = 1 << k;
        let mut new_state = HashMap::new();
        for (&key, &amp) in &self.state {
            if (key & mask) == 0 {
                new_state.insert(key | mask, (-amp.1, amp.0));
            } else {
                new_state.insert(key & !mask, (amp.1, -amp.0));
            }
        }
        self.state = new_state;
    }

    pub fn apply_z(&mut self, k: usize) {
        let mask = 1 << k;
        for (key, amp) in &mut self.state {
            if (*key & mask) != 0 {
                amp.0 = -amp.0;
                amp.1 = -amp.1;
            }
        }
    }

    pub fn apply_s(&mut self, k: usize) {
        let mask = 1 << k;
        for (key, amp) in &mut self.state {
            if (*key & mask) != 0 {
                let temp = amp.0;
                amp.0 = -amp.1;
                amp.1 = temp;
            }
        }
    }

    pub fn apply_t(&mut self, k: usize) {
        let mask = 1 << k;
        let cos_t = FRAC_1_SQRT_2;
        let sin_t = FRAC_1_SQRT_2;
        for (key, amp) in &mut self.state {
            if (*key & mask) != 0 {
                let (re, im) = *amp;
                *amp = (re * cos_t - im * sin_t, re * sin_t + im * cos_t);
            }
        }
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        let mut new_state = HashMap::new();
        for (key, amp) in &self.state {
            if (key & c_mask) != 0 {
                new_state.insert(key ^ t_mask, *amp);
            } else {
                new_state.insert(*key, *amp);
            }
        }
        self.state = new_state;
    }

    pub fn apply_cz(&mut self, control: usize, target: usize) {
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        for (key, amp) in &mut self.state {
            if (*key & c_mask) != 0 && (*key & t_mask) != 0 {
                amp.0 = -amp.0;
                amp.1 = -amp.1;
            }
        }
    }

    pub fn apply_swap(&mut self, q1: usize, q2: usize) {
        let mask1 = 1 << q1;
        let mask2 = 1 << q2;
        let mut new_state = HashMap::new();
        for (key, amp) in &self.state {
            let b1 = (key & mask1) != 0;
            let b2 = (key & mask2) != 0;
            if b1 != b2 {
                new_state.insert(key ^ (mask1 | mask2), *amp);
            } else {
                new_state.insert(*key, *amp);
            }
        }
        self.state = new_state;
    }

    pub fn apply_ccx(&mut self, control1: usize, control2: usize, target: usize) {
        let c1_mask = 1 << control1;
        let c2_mask = 1 << control2;
        let t_mask = 1 << target;
        let mut new_state = HashMap::new();
        for (key, amp) in &self.state {
            if (key & c1_mask) != 0 && (key & c2_mask) != 0 {
                new_state.insert(key ^ t_mask, *amp);
            } else {
                new_state.insert(*key, *amp);
            }
        }
        self.state = new_state;
    }

    pub fn apply_cswap(&mut self, control: usize, q1: usize, q2: usize) {
        let c_mask = 1 << control;
        let mask1 = 1 << q1;
        let mask2 = 1 << q2;
        let mut new_state = HashMap::new();
        for (key, amp) in &self.state {
            if (key & c_mask) != 0 {
                let b1 = (key & mask1) != 0;
                let b2 = (key & mask2) != 0;
                if b1 != b2 {
                    new_state.insert(key ^ (mask1 | mask2), *amp);
                } else {
                    new_state.insert(*key, *amp);
                }
            } else {
                new_state.insert(*key, *amp);
            }
        }
        self.state = new_state;
    }

    pub fn apply_cp(&mut self, control: usize, target: usize, theta: f64) {
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        let cos_val = theta.cos();
        let sin_val = theta.sin();
        for (key, amp) in &mut self.state {
            if (*key & c_mask) != 0 && (*key & t_mask) != 0 {
                let (re, im) = *amp;
                *amp = (re * cos_val - im * sin_val, re * sin_val + im * cos_val);
            }
        }
    }

    pub fn apply_crx(&mut self, control: usize, target: usize, theta: f64) {
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        let mut groups: HashMap<u64, [(f64, f64); 2]> = HashMap::new();
        let mut unchanged_state = HashMap::new();
        for (&key, &amp) in &self.state {
            if (key & c_mask) != 0 {
                let base = key & !t_mask;
                let bit = (key & t_mask) != 0;
                let entry = groups.entry(base).or_insert([(0.0, 0.0), (0.0, 0.0)]);
                if bit {
                    entry[1] = amp;
                } else {
                    entry[0] = amp;
                }
            } else {
                unchanged_state.insert(key, amp);
            }
        }
        for (base, [a0, a1]) in groups {
            let v0 = (
                cos_val * a0.0 + sin_val * a1.1,
                cos_val * a0.1 - sin_val * a1.0,
            );
            let v1 = (
                sin_val * a0.1 + cos_val * a1.0,
                -sin_val * a0.0 + cos_val * a1.1,
            );
            if v0.0.powi(2) + v0.1.powi(2) > 1e-24 {
                unchanged_state.insert(base, v0);
            }
            if v1.0.powi(2) + v1.1.powi(2) > 1e-24 {
                unchanged_state.insert(base | t_mask, v1);
            }
        }
        self.state = unchanged_state;
    }

    pub fn apply_cry(&mut self, control: usize, target: usize, theta: f64) {
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        let mut groups: HashMap<u64, [(f64, f64); 2]> = HashMap::new();
        let mut unchanged_state = HashMap::new();
        for (&key, &amp) in &self.state {
            if (key & c_mask) != 0 {
                let base = key & !t_mask;
                let bit = (key & t_mask) != 0;
                let entry = groups.entry(base).or_insert([(0.0, 0.0), (0.0, 0.0)]);
                if bit {
                    entry[1] = amp;
                } else {
                    entry[0] = amp;
                }
            } else {
                unchanged_state.insert(key, amp);
            }
        }
        for (base, [a0, a1]) in groups {
            let v0 = (
                cos_val * a0.0 - sin_val * a1.0,
                cos_val * a0.1 - sin_val * a1.1,
            );
            let v1 = (
                sin_val * a0.0 + cos_val * a1.0,
                sin_val * a0.1 + cos_val * a1.1,
            );
            if v0.0.powi(2) + v0.1.powi(2) > 1e-24 {
                unchanged_state.insert(base, v0);
            }
            if v1.0.powi(2) + v1.1.powi(2) > 1e-24 {
                unchanged_state.insert(base | t_mask, v1);
            }
        }
        self.state = unchanged_state;
    }

    pub fn apply_crz(&mut self, control: usize, target: usize, theta: f64) {
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        let cos_val = (theta / 2.0).cos();
        let sin_val = (theta / 2.0).sin();
        for (key, amp) in &mut self.state {
            if (*key & c_mask) != 0 {
                let has_target = (*key & t_mask) != 0;
                let (re, im) = *amp;
                if !has_target {
                    *amp = (re * cos_val + im * sin_val, im * cos_val - re * sin_val);
                } else {
                    *amp = (re * cos_val - im * sin_val, im * cos_val + re * sin_val);
                }
            }
        }
    }

    pub fn measure(&mut self, k: usize, r: f64) -> usize {
        let mask = 1 << k;
        let mut p0 = 0.0;
        for (key, amp) in &self.state {
            if (key & mask) == 0 {
                p0 += amp.0.powi(2) + amp.1.powi(2);
            }
        }
        let outcome = if r < p0 {
            let norm = p0.sqrt();
            let norm_val = if norm > 1e-15 { norm } else { 1.0 };
            let mut new_state = HashMap::new();
            for (key, amp) in &self.state {
                if (key & mask) == 0 {
                    new_state.insert(*key, (amp.0 / norm_val, amp.1 / norm_val));
                }
            }
            self.state = new_state;
            0
        } else {
            let p1 = 1.0 - p0;
            let norm = p1.sqrt();
            let norm_val = if norm > 1e-15 { norm } else { 1.0 };
            let mut new_state = HashMap::new();
            for (key, amp) in &self.state {
                if (key & mask) != 0 {
                    new_state.insert(*key, (amp.0 / norm_val, amp.1 / norm_val));
                }
            }
            self.state = new_state;
            1
        };
        outcome
    }

    pub fn get_state_vector(&self) -> Vec<(f64, f64)> {
        let n = 1 << self.num_qubits;
        let mut vec = vec![(0.0, 0.0); n];
        for (&key, &amp) in &self.state {
            if (key as usize) < n {
                vec[key as usize] = amp;
            }
        }
        vec
    }

    pub fn get_state_list(&self) -> Vec<(u64, (f64, f64))> {
        self.state.iter().map(|(&k, &v)| (k, v)).collect()
    }

    pub fn set_state_list(&mut self, state_list: Vec<(u64, (f64, f64))>) {
        self.state = state_list.into_iter().collect();
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RustSparseSimulator {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[pyo3(name = "allocate_qubit")]
    pub fn allocate_qubit_py(&mut self) {
        self.allocate_qubit();
    }

    #[pyo3(name = "apply_1qubit_gate")]
    pub fn apply_1qubit_gate_py(
        &mut self,
        k: usize,
        u00_re: f64, u00_im: f64,
        u01_re: f64, u01_im: f64,
        u10_re: f64, u10_im: f64,
        u11_re: f64, u11_im: f64,
    ) {
        self.apply_1qubit_gate(k, u00_re, u00_im, u01_re, u01_im, u10_re, u10_im, u11_re, u11_im);
    }

    #[pyo3(name = "apply_h")]
    pub fn apply_h_py(&mut self, k: usize) {
        self.apply_h(k);
    }

    #[pyo3(name = "apply_x")]
    pub fn apply_x_py(&mut self, k: usize) {
        self.apply_x(k);
    }

    #[pyo3(name = "apply_y")]
    pub fn apply_y_py(&mut self, k: usize) {
        self.apply_y(k);
    }

    #[pyo3(name = "apply_z")]
    pub fn apply_z_py(&mut self, k: usize) {
        self.apply_z(k);
    }

    #[pyo3(name = "apply_s")]
    pub fn apply_s_py(&mut self, k: usize) {
        self.apply_s(k);
    }

    #[pyo3(name = "apply_t")]
    pub fn apply_t_py(&mut self, k: usize) {
        self.apply_t(k);
    }

    #[pyo3(name = "apply_cnot")]
    pub fn apply_cnot_py(&mut self, control: usize, target: usize) {
        self.apply_cnot(control, target);
    }

    #[pyo3(name = "apply_cz")]
    pub fn apply_cz_py(&mut self, control: usize, target: usize) {
        self.apply_cz(control, target);
    }

    #[pyo3(name = "apply_swap")]
    pub fn apply_swap_py(&mut self, q1: usize, q2: usize) {
        self.apply_swap(q1, q2);
    }

    #[pyo3(name = "apply_ccx")]
    pub fn apply_ccx_py(&mut self, control1: usize, control2: usize, target: usize) {
        self.apply_ccx(control1, control2, target);
    }

    #[pyo3(name = "apply_cswap")]
    pub fn apply_cswap_py(&mut self, control: usize, q1: usize, q2: usize) {
        self.apply_cswap(control, q1, q2);
    }

    #[pyo3(name = "apply_cp")]
    pub fn apply_cp_py(&mut self, control: usize, target: usize, theta: f64) {
        self.apply_cp(control, target, theta);
    }

    #[pyo3(name = "apply_crx")]
    pub fn apply_crx_py(&mut self, control: usize, target: usize, theta: f64) {
        self.apply_crx(control, target, theta);
    }

    #[pyo3(name = "apply_cry")]
    pub fn apply_cry_py(&mut self, control: usize, target: usize, theta: f64) {
        self.apply_cry(control, target, theta);
    }

    #[pyo3(name = "apply_crz")]
    pub fn apply_crz_py(&mut self, control: usize, target: usize, theta: f64) {
        self.apply_crz(control, target, theta);
    }

    #[pyo3(name = "measure")]
    pub fn measure_py(&mut self, k: usize, r: f64) -> usize {
        self.measure(k, r)
    }

    #[pyo3(name = "get_state_vector")]
    pub fn get_state_vector_py(&self) -> Vec<(f64, f64)> {
        self.get_state_vector()
    }

    #[pyo3(name = "get_state_list")]
    pub fn get_state_list_py(&self) -> Vec<(u64, (f64, f64))> {
        self.get_state_list()
    }

    #[pyo3(name = "set_state_list")]
    pub fn set_state_list_py(&mut self, state_list: Vec<(u64, (f64, f64))>) {
        self.set_state_list(state_list);
    }
}

fn complex_svd(
    a: &[Vec<(f64, f64)>],
) -> (
    Vec<Vec<(f64, f64)>>,
    Vec<f64>,
    Vec<Vec<(f64, f64)>>,
) {
    let rows = a.len();
    let cols = a[0].len();
    
    let mut columns = vec![vec![(0.0, 0.0); rows]; cols];
    for r in 0..rows {
        for c in 0..cols {
            columns[c][r] = a[r][c];
        }
    }
    
    let mut v = vec![vec![(0.0, 0.0); cols]; cols];
    for i in 0..cols {
        v[i][i] = (1.0, 0.0);
    }
    
    let max_sweeps = 30;
    let eps = 1e-12;
    
    for _sweep in 0..max_sweeps {
        let mut converged = true;
        for i in 0..cols {
            for j in (i+1)..cols {
                let mut dot_ii = 0.0;
                let mut dot_jj = 0.0;
                let mut dot_ij = (0.0, 0.0);
                
                for r in 0..rows {
                    let ci = columns[i][r];
                    let cj = columns[j][r];
                    dot_ii += ci.0 * ci.0 + ci.1 * ci.1;
                    dot_jj += cj.0 * cj.0 + cj.1 * cj.1;
                    dot_ij.0 += ci.0 * cj.0 + ci.1 * cj.1;
                    dot_ij.1 += ci.1 * cj.0 - ci.0 * cj.1;
                }
                
                let r = (dot_ij.0.powi(2) + dot_ij.1.powi(2)).sqrt();
                if r < eps * (dot_ii * dot_jj).sqrt() {
                    continue;
                }
                converged = false;
                
                let phi = dot_ij.1.atan2(dot_ij.0);
                let theta = 0.5 * (2.0 * r).atan2(dot_jj - dot_ii);
                
                let c = theta.cos();
                let s = theta.sin();
                
                let s_phase = (s * phi.cos(), s * phi.sin());
                let s_phase_conj = (s * phi.cos(), -s * phi.sin());
                
                for r in 0..rows {
                    let ci = columns[i][r];
                    let cj = columns[j][r];
                    
                    let new_ci = (
                        c * ci.0 + (s_phase.0 * cj.0 - s_phase.1 * cj.1),
                        c * ci.1 + (s_phase.0 * cj.1 + s_phase.1 * cj.0),
                    );
                    let new_cj = (
                        - (s_phase_conj.0 * ci.0 - s_phase_conj.1 * ci.1) + c * cj.0,
                        - (s_phase_conj.0 * ci.1 + s_phase_conj.1 * ci.0) + c * cj.1,
                    );
                    
                    columns[i][r] = new_ci;
                    columns[j][r] = new_cj;
                }
                
                for r in 0..cols {
                    let vi = v[i][r];
                    let vj = v[j][r];
                    
                    let new_vi = (
                        c * vi.0 + (s_phase.0 * vj.0 - s_phase.1 * vj.1),
                        c * vi.1 + (s_phase.0 * vj.1 + s_phase.1 * vj.0),
                    );
                    let new_vj = (
                        - (s_phase_conj.0 * vi.0 - s_phase_conj.1 * vi.1) + c * vj.0,
                        - (s_phase_conj.0 * vi.1 + s_phase_conj.1 * vi.0) + c * vj.1,
                    );
                    
                    v[i][r] = new_vi;
                    v[j][r] = new_vj;
                }
            }
        }
        if converged {
            break;
        }
    }
    
    let mut s = vec![0.0; cols];
    let mut u = vec![vec![(0.0, 0.0); cols]; rows];
    
    for c in 0..cols {
        let mut norm2 = 0.0;
        for r in 0..rows {
            let val = columns[c][r];
            norm2 += val.0 * val.0 + val.1 * val.1;
        }
        let norm = norm2.sqrt();
        s[c] = norm;
        
        let norm_val = if norm > 1e-15 { norm } else { 1.0 };
        for r in 0..rows {
            u[r][c] = (columns[c][r].0 / norm_val, columns[c][r].1 / norm_val);
        }
    }
    
    let mut vh = vec![vec![(0.0, 0.0); cols]; cols];
    for r in 0..cols {
        for c in 0..cols {
            let val = v[r][c];
            vh[r][c] = (val.0, -val.1);
        }
    }
    
    (u, s, vh)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn compute_svd_native(matrix: Vec<Vec<(f64, f64)>>) -> PyResult<(Vec<Vec<(f64, f64)>>, Vec<f64>, Vec<Vec<(f64, f64)>>)> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let (u, s, vh) = complex_svd(&matrix);
    Ok((u, s, vh))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_sv_zero_copy() {
        let mut sv = RustStatevector::new();
        sv.allocate_qubit().unwrap();
        sv.allocate_qubit().unwrap();
        let ptr_before = sv.state.as_ptr();
        sv.apply_h(0).unwrap();
        sv.apply_x(1).unwrap();
        let ptr_after = sv.state.as_ptr();
        assert_eq!(ptr_before, ptr_after, "Statevector buffer was copied or reallocated during gate application!");
    }
}
