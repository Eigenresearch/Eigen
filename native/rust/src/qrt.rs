use crate::simulator::RustStatevector;
use std::os::raw::c_char;
use std::ffi::CStr;

pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        // LCG constants from Numerical Recipes
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [0, 1) float
        (self.state as f64) / (u64::MAX as f64)
    }
}

pub struct QrtSimulator {
    pub sv: RustStatevector,
    pub rng: LcgRng,
}

#[no_mangle]
pub extern "C" fn eigen_qrt_init(mut seed: u64) -> *mut QrtSimulator {
    if seed == 0 {
        if let Ok(env_seed_str) = std::env::var("EIGEN_SEED") {
            if let Ok(env_seed) = env_seed_str.parse::<u64>() {
                seed = env_seed;
            }
        }
    }
    if seed == 0 {
        use std::time::SystemTime;
        seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
    }
    let qrt = Box::new(QrtSimulator {
        sv: RustStatevector::new(),
        rng: LcgRng::new(seed),
    });
    Box::into_raw(qrt)
}

#[no_mangle]
pub extern "C" fn eigen_qrt_alloc(sim: *mut QrtSimulator) -> i32 {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.allocate_qubit();
    // Return index of newly allocated qubit (num_qubits - 1)
    let n = sim.sv.get_state().len();
    (n.trailing_zeros() as i32) - 1
}

#[no_mangle]
pub extern "C" fn eigen_qrt_h(sim: *mut QrtSimulator, qubit: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_h(qubit as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_x(sim: *mut QrtSimulator, qubit: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_x(qubit as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_y(sim: *mut QrtSimulator, qubit: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_y(qubit as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_z(sim: *mut QrtSimulator, qubit: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_z(qubit as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_s(sim: *mut QrtSimulator, qubit: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_s(qubit as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_t(sim: *mut QrtSimulator, qubit: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_t(qubit as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_rx(sim: *mut QrtSimulator, qubit: i32, theta: f64) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_rx(qubit as usize, theta);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_ry(sim: *mut QrtSimulator, qubit: i32, theta: f64) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_ry(qubit as usize, theta);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_rz(sim: *mut QrtSimulator, qubit: i32, theta: f64) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_rz(qubit as usize, theta);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_cnot(sim: *mut QrtSimulator, control: i32, target: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_cnot(control as usize, target as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_cz(sim: *mut QrtSimulator, control: i32, target: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_cz(control as usize, target as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_swap(sim: *mut QrtSimulator, q1: i32, q2: i32) {
    let sim = unsafe { &mut *sim };
    let _ = sim.sv.apply_swap(q1 as usize, q2 as usize);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_measure(sim: *mut QrtSimulator, qubit: i32) -> i32 {
    let sim = unsafe { &mut *sim };
    let r = sim.rng.next_f64();
    sim.sv.measure(qubit as usize, r).unwrap_or(0) as i32
}

#[no_mangle]
pub extern "C" fn eigen_qrt_trace(sim: *mut QrtSimulator) {
    let sim = unsafe { &mut *sim };
    let state = sim.sv.get_state();
    let num_qubits = state.len().trailing_zeros() as usize;
    
    let mut parts = Vec::new();
    for i in 0..state.len() {
        let (real, imag) = state[i];
        let prob = real * real + imag * imag;
        if prob > 1e-12 {
            let mut bitstring = String::new();
            for q in (0..num_qubits).rev() {
                bitstring.push(if ((i >> q) & 1) != 0 { '1' } else { '0' });
            }
            
            let amp_str = if imag.abs() < 1e-9 {
                format!("{:.5}", real)
            } else if real.abs() < 1e-9 {
                format!("{:.5}i", imag)
            } else {
                let sign = if imag >= 0.0 { "+" } else { "-" };
                format!("({:.5} {} {:.5}i)", real, sign, imag.abs())
            };
            parts.push(format!("{} * |{}> (prob={:.1}%)", amp_str, bitstring, prob * 100.0));
        }
    }
    println!("Quantum State: {}", parts.join(" + "));
}

#[no_mangle]
pub extern "C" fn eigen_qrt_free(sim: *mut QrtSimulator) {
    if !sim.is_null() {
        unsafe {
            let _ = Box::from_raw(sim);
        }
    }
}

// Formatters matching Python's string outputs
#[no_mangle]
pub extern "C" fn eigen_qrt_print_int(val: i64) {
    println!("{}", val);
}

#[no_mangle]
pub extern "C" fn eigen_qrt_print_bool(val: bool) {
    println!("{}", if val { "True" } else { "False" });
}

#[no_mangle]
pub extern "C" fn eigen_qrt_print_float(val: f64) {
    if val.is_nan() {
        println!("nan");
        return;
    }
    if val == f64::INFINITY {
        println!("inf");
        return;
    }
    if val == f64::NEG_INFINITY {
        println!("-inf");
        return;
    }

    let mut s = format!("{}", val);
    if let Some(e_idx) = s.find(|c| c == 'e' || c == 'E') {
        let significand = &s[..e_idx];
        let exponent_str = &s[e_idx + 1..];
        if let Ok(exp) = exponent_str.parse::<i32>() {
            if exp >= 0 {
                println!("{}e+{:02}", significand, exp);
            } else {
                println!("{}e-{:02}", significand, -exp);
            }
        } else {
            println!("{}", s);
        }
    } else {
        if !s.contains('.') {
            s.push_str(".0");
        }
        println!("{}", s);
    }
}

#[no_mangle]
pub extern "C" fn eigen_qrt_print_string(val: *const c_char) {
    if !val.is_null() {
        let c_str = unsafe { CStr::from_ptr(val) };
        if let Ok(s) = c_str.to_str() {
            println!("{}", s);
        }
    }
}

#[no_mangle]
pub extern "C" fn eigen_qrt_panic_div_zero() {
    eprintln!("ZeroDivisionError: division by zero");
    std::process::exit(1);
}
