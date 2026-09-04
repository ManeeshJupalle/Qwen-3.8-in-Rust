//! CPU topology: how many *physical* cores this machine has, which is the engine's default thread count.
//!
//! The matvec kernels are memory-bound and every participant streams its own weight rows, so two hardware
//! threads sharing one core's load ports and L1 do not add bandwidth; they add contention. On the i7-9750H
//! (6 cores / 12 threads) 6 threads match or beat 12 on every kernel line and every membw line
//! (`docs/data/kernels_bench.txt`), so `aqueduct run` and `bench` default to the physical count and
//! `--threads N` overrides it.
//!
//! Windows: `GetLogicalProcessorInformationEx(RelationProcessorCore, ..)` returns one variable-length record
//! per physical core; the count of records is the answer. Linux: the number of distinct
//! `thread_siblings_list` values under `/sys/devices/system/cpu/cpu*/topology` (the SMT siblings of one core
//! all report the same list). Anything else, or any failure: `available_parallelism`.

/// Hardware threads (SMT siblings counted separately); 1 if the platform will not say.
pub fn logical_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[cfg(windows)]
pub fn physical_cores() -> usize {
    const RELATION_PROCESSOR_CORE: u32 = 0;
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
    extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: u32, buffer: *mut u8, returned_length: *mut u32) -> i32;
        fn GetLastError() -> u32;
    }
    let mut len: u32 = 0;
    // SAFETY: the documented size query: a null buffer and a zero length, which fails with
    // ERROR_INSUFFICIENT_BUFFER and writes the required byte count into `len`.
    let (ok, err) = unsafe { (GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len), GetLastError()) };
    if ok != 0 || err != ERROR_INSUFFICIENT_BUFFER || len == 0 {
        return logical_cores();
    }
    let mut buf = vec![0u8; len as usize];
    // SAFETY: `buf` holds exactly the `len` bytes the size query asked for.
    if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len) } == 0 {
        return logical_cores();
    }
    // Each record is `u32 Relationship, u32 Size, <union>`; `Size` walks to the next one. The query filtered
    // to RelationProcessorCore, so every record is one physical core (the relationship is re-checked anyway).
    let total = (len as usize).min(buf.len());
    let mut off = 0usize;
    let mut cores = 0usize;
    while off + 8 <= total {
        let relationship = u32::from_ne_bytes(buf[off..off + 4].try_into().unwrap());
        let size = u32::from_ne_bytes(buf[off + 4..off + 8].try_into().unwrap()) as usize;
        if size < 8 || off + size > total {
            break;
        }
        if relationship == RELATION_PROCESSOR_CORE {
            cores += 1;
        }
        off += size;
    }
    if cores == 0 {
        logical_cores()
    } else {
        cores.min(logical_cores())
    }
}

#[cfg(target_os = "linux")]
pub fn physical_cores() -> usize {
    use std::collections::BTreeSet;
    let dir = match std::fs::read_dir("/sys/devices/system/cpu") {
        Ok(d) => d,
        Err(_) => return logical_cores(),
    };
    let mut groups: BTreeSet<String> = BTreeSet::new();
    for e in dir.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) || name.len() == 3 {
            continue;
        }
        // the SMT siblings of one core all report the same list, so one entry per physical core survives
        if let Ok(s) = std::fs::read_to_string(e.path().join("topology/thread_siblings_list")) {
            groups.insert(s.trim().to_string());
        }
    }
    if groups.is_empty() {
        logical_cores()
    } else {
        groups.len().min(logical_cores())
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn physical_cores() -> usize {
    logical_cores()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_cores_is_sane() {
        let (p, l) = (physical_cores(), logical_cores());
        assert!(p >= 1 && p <= l, "physical {p} logical {l}");
        // SMT gives at most 2 threads per core on every CPU this engine targets
        assert!(l <= 2 * p, "logical {l} is more than twice physical {p}");
        println!("cpu topology: {p} physical cores, {l} logical");
    }
}
