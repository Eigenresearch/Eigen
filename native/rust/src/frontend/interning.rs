use std::sync::{OnceLock, Mutex};
use std::collections::HashSet;

static INTERNER: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Interns a string slice, returning a static reference to a single shared allocation.
pub fn intern(s: &str) -> &'static str {
    let mutex = INTERNER.get_or_init(|| Mutex::new(HashSet::new()));
    let mut set = mutex.lock().unwrap();
    if let Some(existing) = set.get(s) {
        unsafe { &*(existing.as_str() as *const str) }
    } else {
        let owned = s.to_string();
        let ptr = owned.as_str() as *const str;
        set.insert(owned);
        unsafe { &*ptr }
    }
}
