//! Platform layer for the tiers (Phase 4): unbuffered file reads, sector-aligned buffers with an optional
//! large-page attempt, the process memory cap, and memory status. Windows through kernel32 / advapi32
//! declarations (no crate); Linux through libc symbols. Only the Windows path has been run; the Linux path
//! compiles (`cargo check --target x86_64-unknown-linux-gnu`) and is marked untested.
//!
//! **Alignment.** An unbuffered read (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) needs the file offset, the byte
//! count and the buffer address to be multiples of the sector size. `DirectFile::sector` is queried from the
//! volume the file lives on (`FILE_STORAGE_INFO::PhysicalBytesPerSectorForPerformance` on Windows, the block
//! device's `logical_block_size` under sysfs on Linux) and never assumed; it is floored at the 4 KiB page so
//! any page-aligned buffer qualifies. Callers read the aligned superset of the span they want and slice.
//!
//! **Queue depth.** A read is issued as `chunk`-byte overlapped requests with up to `qd` in flight; qd 1 is
//! one request at a time. `aqueduct doctor` measures both on the model's drive with this very function.

use std::io;
use std::path::Path;
use std::sync::OnceLock;

/// Round `x` down / up to a multiple of `a` (a power of two or not; `a > 0`).
pub fn align_down(x: u64, a: u64) -> u64 {
    x - x % a
}
pub fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// Bytes an arena needs to hold the aligned superset of any `span`-byte range: `align_up(span) + sector`
/// covers every possible start offset modulo the sector (the superset is at most `span + 2 * (sector - 1)`
/// rounded up to a sector multiple).
pub fn arena_bytes(span: u64, sector: u64) -> u64 {
    align_up(span, sector) + sector
}

/// Read chunk for the overlapped requests (each request is at most this long).
pub const DEFAULT_CHUNK: usize = 32 << 20;

/// Why large pages were not used, if they were not (set once, the first time an arena falls back).
static LARGE_PAGE_NOTE: OnceLock<String> = OnceLock::new();

/// The large-page fallback reason, if any arena fell back so far.
pub fn large_page_note() -> Option<&'static str> {
    LARGE_PAGE_NOTE.get().map(String::as_str)
}

fn note_large_page_fallback(reason: String) {
    if LARGE_PAGE_NOTE.set(reason.clone()).is_ok() {
        eprintln!("large pages: {reason}; using normal pages");
    }
}

/// A page- (or large-page-) aligned, committed buffer. Handed out as raw pointers: the ring protocol in
/// `tier.rs` is what makes a slice of it valid (the I/O thread writes a slot only while no reader holds it).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    /// Whether this buffer got large pages.
    pub large_pages: bool,
}

// SAFETY: the buffer is plain memory; concurrent access is governed by the ring protocol, not by the type.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuf({} bytes at {:p}, large_pages {})", self.len, self.ptr, self.large_pages)
    }
}

impl AlignedBuf {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// `len` bytes from `off`.
    ///
    /// # Safety
    /// The range must lie inside the buffer and nothing may be writing it (see the module doc).
    pub unsafe fn slice(&self, off: usize, len: usize) -> &[u8] {
        debug_assert!(off + len <= self.len);
        std::slice::from_raw_parts(self.ptr.add(off), len)
    }
}

// ================================================================================================ Windows
#[cfg(windows)]
mod imp {
    use super::*;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    type Handle = *mut c_void;
    const INVALID_HANDLE: Handle = -1isize as Handle;
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const OPEN_EXISTING: u32 = 3;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    const ERROR_IO_PENDING: u32 = 997;
    const ERROR_HANDLE_EOF: u32 = 38;
    const ERROR_NOT_ALL_ASSIGNED: u32 = 1300;
    const FILE_STORAGE_INFO_CLASS: u32 = 16;
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_LARGE_PAGES: u32 = 0x2000_0000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 4;
    const TOKEN_ADJUST_PRIVILEGES: u32 = 0x20;
    const TOKEN_QUERY: u32 = 8;
    const SE_PRIVILEGE_ENABLED: u32 = 2;
    const JOB_OBJECT_LIMIT_PROCESS_MEMORY: u32 = 0x100;
    const JOB_OBJECT_LIMIT_JOB_MEMORY: u32 = 0x200;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: u32 = 9;

    #[repr(C)]
    struct Overlapped {
        internal: usize,
        internal_high: usize,
        offset: u32,
        offset_high: u32,
        event: Handle,
    }

    #[repr(C)]
    struct FileStorageInfo {
        logical_bytes_per_sector: u32,
        physical_bytes_per_sector_for_atomicity: u32,
        physical_bytes_per_sector_for_performance: u32,
        file_system_effective_physical_bytes_per_sector_for_atomicity: u32,
        flags: u32,
        byte_offset_for_sector_alignment: u32,
        byte_offset_for_partition_alignment: u32,
    }

    #[repr(C)]
    struct Luid {
        low: u32,
        high: i32,
    }
    #[repr(C)]
    struct TokenPrivileges {
        count: u32,
        luid: Luid,
        attributes: u32,
    }

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct JobBasicLimit {
        per_process_user_time: i64,
        per_job_user_time: i64,
        limit_flags: u32,
        min_working_set: usize,
        max_working_set: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct JobExtendedLimit {
        basic: JobBasicLimit,
        io: [u64; 6],
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(name: *const u16, access: u32, share: u32, sec: *mut c_void, disp: u32, flags: u32, template: Handle) -> Handle;
        fn ReadFile(h: Handle, buf: *mut c_void, n: u32, read: *mut u32, ov: *mut Overlapped) -> i32;
        fn GetOverlappedResult(h: Handle, ov: *mut Overlapped, transferred: *mut u32, wait: i32) -> i32;
        fn CreateEventW(sec: *mut c_void, manual: i32, initial: i32, name: *const u16) -> Handle;
        fn CloseHandle(h: Handle) -> i32;
        fn GetLastError() -> u32;
        fn GetFileInformationByHandleEx(h: Handle, class: u32, info: *mut c_void, size: u32) -> i32;
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
        fn GetLargePageMinimum() -> usize;
        fn GetCurrentProcess() -> Handle;
        fn GlobalMemoryStatusEx(s: *mut MemoryStatusEx) -> i32;
        fn CreateJobObjectW(sec: *mut c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(job: Handle, class: u32, info: *mut c_void, len: u32) -> i32;
        fn QueryInformationJobObject(job: Handle, class: u32, info: *mut c_void, len: u32, ret: *mut u32) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
    }
    #[link(name = "advapi32")]
    extern "system" {
        fn OpenProcessToken(process: Handle, access: u32, token: *mut Handle) -> i32;
        fn LookupPrivilegeValueW(system: *const u16, name: *const u16, luid: *mut Luid) -> i32;
        fn AdjustTokenPrivileges(token: Handle, disable_all: i32, new: *mut TokenPrivileges, len: u32, prev: *mut c_void, ret: *mut u32) -> i32;
    }

    fn wide(s: &str) -> Vec<u16> {
        std::ffi::OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
    }

    fn last_error(what: &str) -> io::Error {
        // SAFETY: plain kernel32 call.
        let code = unsafe { GetLastError() };
        io::Error::new(io::Error::from_raw_os_error(code as i32).kind(), format!("{what}: os error {code}"))
    }

    // ---- large pages ----------------------------------------------------------------------------------

    /// Try to enable SeLockMemoryPrivilege on this process's token (needed for `MEM_LARGE_PAGES`). The
    /// privilege must already be assigned to the user by policy; this only switches it on. The result is
    /// cached: the outcome does not change within a process.
    pub fn try_enable_lock_memory_privilege() -> Result<(), String> {
        static R: OnceLock<Result<(), String>> = OnceLock::new();
        R.get_or_init(|| {
            let mut token: Handle = std::ptr::null_mut();
            // SAFETY: documented advapi32 / kernel32 calls with properly sized structs.
            unsafe {
                if OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut token) == 0 {
                    return Err(format!("OpenProcessToken failed (os error {})", GetLastError()));
                }
                let mut tp = TokenPrivileges { count: 1, luid: Luid { low: 0, high: 0 }, attributes: SE_PRIVILEGE_ENABLED };
                let name = wide("SeLockMemoryPrivilege");
                if LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &mut tp.luid) == 0 {
                    let e = GetLastError();
                    CloseHandle(token);
                    return Err(format!("LookupPrivilegeValue failed (os error {e})"));
                }
                let ok = AdjustTokenPrivileges(token, 0, &mut tp, std::mem::size_of::<TokenPrivileges>() as u32, std::ptr::null_mut(), std::ptr::null_mut());
                let e = GetLastError();
                CloseHandle(token);
                if ok == 0 {
                    return Err(format!("AdjustTokenPrivileges failed (os error {e})"));
                }
                if e == ERROR_NOT_ALL_ASSIGNED {
                    return Err("SeLockMemoryPrivilege is not held by this user (Local Security Policy > Lock pages in memory); not changing policy".into());
                }
                Ok(())
            }
        })
        .clone()
    }

    /// The large-page size (0 when the platform has none).
    pub fn large_page_minimum() -> usize {
        // SAFETY: plain kernel32 call.
        unsafe { GetLargePageMinimum() }
    }

    impl AlignedBuf {
        /// A committed, page-aligned buffer of at least `len` bytes; with `want_large`, large pages are tried
        /// first (privilege permitting) and the fallback is logged once.
        pub fn new(len: usize, want_large: bool) -> io::Result<AlignedBuf> {
            let len = len.max(4096);
            if want_large {
                let lp = large_page_minimum();
                match (lp, try_enable_lock_memory_privilege()) {
                    (0, _) => note_large_page_fallback("this platform reports no large-page size".into()),
                    (_, Err(reason)) => note_large_page_fallback(reason),
                    (lp, Ok(())) => {
                        let rounded = len.div_ceil(lp) * lp;
                        // SAFETY: documented VirtualAlloc; a null return is the failure path handled below.
                        let p = unsafe { VirtualAlloc(std::ptr::null_mut(), rounded, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE) };
                        if !p.is_null() {
                            return Ok(AlignedBuf { ptr: p as *mut u8, len: rounded, large_pages: true });
                        }
                        note_large_page_fallback(format!("VirtualAlloc(MEM_LARGE_PAGES) failed (os error {})", unsafe { GetLastError() }));
                    }
                }
            }
            // SAFETY: documented VirtualAlloc; the result is page-aligned and zero-filled on demand.
            let p = unsafe { VirtualAlloc(std::ptr::null_mut(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
            if p.is_null() {
                return Err(last_error(&format!("VirtualAlloc({len} bytes)")));
            }
            Ok(AlignedBuf { ptr: p as *mut u8, len, large_pages: false })
        }
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            // SAFETY: `ptr` came from VirtualAlloc and is released exactly once.
            unsafe { VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE) };
        }
    }

    // ---- unbuffered reads -----------------------------------------------------------------------------

    /// A file opened for unbuffered overlapped reads.
    pub struct DirectFile {
        h: Handle,
        /// Required alignment of offsets, lengths and buffers (queried, floored at 4096).
        pub sector: usize,
        /// Bytes per overlapped request.
        pub chunk: usize,
        /// Requests in flight at once.
        pub qd: usize,
        pub file_size: u64,
        events: Vec<Handle>,
        ovs: Vec<Overlapped>,
        inflight: Vec<(usize, u32)>,
    }

    // SAFETY: the handle and events are only ever used from the thread that owns the struct.
    unsafe impl Send for DirectFile {}

    impl DirectFile {
        pub fn open(path: &Path, chunk: usize, qd: usize) -> io::Result<DirectFile> {
            let qd = qd.max(1);
            let name = wide(&path.to_string_lossy());
            // SAFETY: documented kernel32 calls.
            let h = unsafe { CreateFileW(name.as_ptr(), GENERIC_READ, FILE_SHARE_READ, std::ptr::null_mut(), OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, std::ptr::null_mut()) };
            if h == INVALID_HANDLE {
                return Err(last_error(&format!("CreateFileW({})", path.display())));
            }
            let mut info = FileStorageInfo { logical_bytes_per_sector: 0, physical_bytes_per_sector_for_atomicity: 0, physical_bytes_per_sector_for_performance: 0, file_system_effective_physical_bytes_per_sector_for_atomicity: 0, flags: 0, byte_offset_for_sector_alignment: 0, byte_offset_for_partition_alignment: 0 };
            // SAFETY: the struct matches FILE_STORAGE_INFO and its size is passed.
            let ok = unsafe { GetFileInformationByHandleEx(h, FILE_STORAGE_INFO_CLASS, &mut info as *mut _ as *mut c_void, std::mem::size_of::<FileStorageInfo>() as u32) };
            if ok == 0 {
                let e = last_error("GetFileInformationByHandleEx(FileStorageInfo)");
                // SAFETY: closing the handle we opened.
                unsafe { CloseHandle(h) };
                return Err(e);
            }
            let sector = (info.physical_bytes_per_sector_for_performance.max(info.logical_bytes_per_sector) as usize).max(4096);
            let file_size = std::fs::metadata(path)?.len();
            let mut events = Vec::with_capacity(qd);
            for _ in 0..qd {
                // SAFETY: documented kernel32 call; manual-reset event, initially unsignalled.
                let e = unsafe { CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null()) };
                if e.is_null() {
                    return Err(last_error("CreateEventW"));
                }
                events.push(e);
            }
            let ovs = events.iter().map(|&e| Overlapped { internal: 0, internal_high: 0, offset: 0, offset_high: 0, event: e }).collect();
            let chunk = align_up(chunk.max(sector) as u64, sector as u64) as usize;
            Ok(DirectFile { h, sector, chunk, qd, file_size, events, ovs, inflight: Vec::with_capacity(qd) })
        }

        /// Read `len` bytes at `offset` into the start of `buf`. Offset and length must be sector multiples
        /// (the buffer is page-aligned by construction). Returns the bytes transferred: `len`, or fewer only
        /// when the range runs past the end of the file. Allocation-free after `open` (the in-flight table is
        /// preallocated), so the ring's I/O thread stays inside the counting-allocator test's zero. The
        /// buffer is written through its raw pointer: with a ring slot the caller (the ring protocol)
        /// guarantees no reader holds the slot while this runs.
        pub fn read_at(&mut self, offset: u64, buf: &AlignedBuf, len: usize) -> io::Result<usize> {
            assert!(len <= buf.len(), "DirectFile::read_at: {len} bytes into a {}-byte buffer", buf.len());
            let buf = buf.as_mut_ptr();
            assert!(offset.is_multiple_of(self.sector as u64) && len.is_multiple_of(self.sector) && (buf as usize).is_multiple_of(self.sector), "DirectFile::read_at: unaligned (offset {offset}, len {len}, buf {buf:p}, sector {})", self.sector);
            self.inflight.clear();
            let mut issued = 0usize; // bytes issued so far
            let mut total = 0usize;
            let mut eof = false;
            let n_slots = self.qd;
            let mut slot_busy = [false; 64];
            assert!(n_slots <= 64, "queue depth above 64 is not supported");
            loop {
                // issue while there is room and data left
                while self.inflight.len() < n_slots && issued < len && !eof {
                    let s = (0..n_slots).find(|&s| !slot_busy[s]).expect("a free slot");
                    let this = (len - issued).min(self.chunk);
                    let off = offset + issued as u64;
                    let ov = &mut self.ovs[s];
                    ov.internal = 0;
                    ov.internal_high = 0;
                    ov.offset = off as u32;
                    ov.offset_high = (off >> 32) as u32;
                    let mut got: u32 = 0;
                    // SAFETY: the buffer range is inside the caller's buffer; the OVERLAPPED and its event
                    // outlive the request (we wait for every request before returning).
                    let ok = unsafe { ReadFile(self.h, buf.add(issued) as *mut c_void, this as u32, &mut got, ov) };
                    if ok != 0 {
                        // completed synchronously
                        total += got as usize;
                        if (got as usize) < this {
                            eof = true;
                        }
                        issued += this;
                        continue;
                    }
                    // SAFETY: plain kernel32 call.
                    let e = unsafe { GetLastError() };
                    if e == ERROR_IO_PENDING {
                        slot_busy[s] = true;
                        self.inflight.push((s, this as u32));
                        issued += this;
                    } else if e == ERROR_HANDLE_EOF {
                        eof = true;
                        issued += this;
                    } else {
                        // drain what is in flight before failing
                        for &(s2, _) in &self.inflight {
                            let mut t: u32 = 0;
                            // SAFETY: as above.
                            unsafe { GetOverlappedResult(self.h, &mut self.ovs[s2], &mut t, 1) };
                        }
                        self.inflight.clear();
                        return Err(io::Error::new(io::Error::from_raw_os_error(e as i32).kind(), format!("ReadFile at {off} ({this} bytes): os error {e}")));
                    }
                }
                if self.inflight.is_empty() {
                    break;
                }
                // wait for the oldest request
                let (s, want) = self.inflight.remove(0);
                slot_busy[s] = false;
                let mut t: u32 = 0;
                // SAFETY: the OVERLAPPED belongs to a request issued above on this handle.
                let ok = unsafe { GetOverlappedResult(self.h, &mut self.ovs[s], &mut t, 1) };
                if ok == 0 {
                    // SAFETY: plain kernel32 call.
                    let e = unsafe { GetLastError() };
                    if e != ERROR_HANDLE_EOF {
                        for &(s2, _) in &self.inflight {
                            let mut t2: u32 = 0;
                            // SAFETY: as above.
                            unsafe { GetOverlappedResult(self.h, &mut self.ovs[s2], &mut t2, 1) };
                        }
                        self.inflight.clear();
                        return Err(io::Error::new(io::Error::from_raw_os_error(e as i32).kind(), format!("GetOverlappedResult: os error {e}")));
                    }
                    eof = true;
                }
                total += t as usize;
                if t < want {
                    eof = true;
                }
            }
            Ok(total)
        }
    }

    impl Drop for DirectFile {
        fn drop(&mut self) {
            // SAFETY: handles we created, closed once.
            unsafe {
                for &e in &self.events {
                    CloseHandle(e);
                }
                CloseHandle(self.h);
            }
        }
    }

    // ---- memory status and the job-object cap ---------------------------------------------------------

    /// (total physical bytes, available physical bytes).
    pub fn mem_status() -> Option<(u64, u64)> {
        let mut s = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, memory_load: 0, total_phys: 0, avail_phys: 0, total_page_file: 0, avail_page_file: 0, total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0 };
        // SAFETY: the struct is sized and its length field set.
        if unsafe { GlobalMemoryStatusEx(&mut s) } == 0 {
            return None;
        }
        Some((s.total_phys, s.avail_phys))
    }

    static JOB: OnceLock<usize> = OnceLock::new();

    /// Put this process into a job object whose committed memory (job-wide and per process) is capped at
    /// `bytes`. From here on an allocation past the cap fails, which Rust turns into an abort: the engine
    /// cannot exceed the cap, it can only die trying. That is the enforcement the ladder relies on; the plan
    /// is what keeps it from ever happening.
    pub fn apply_job_memory_limit(bytes: u64) -> io::Result<()> {
        // SAFETY: documented kernel32 calls with a correctly sized struct.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if job.is_null() {
                return Err(last_error("CreateJobObjectW"));
            }
            let mut ext = JobExtendedLimit::default();
            ext.basic.limit_flags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            ext.process_memory_limit = bytes as usize;
            ext.job_memory_limit = bytes as usize;
            if SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, &mut ext as *mut _ as *mut c_void, std::mem::size_of::<JobExtendedLimit>() as u32) == 0 {
                return Err(last_error("SetInformationJobObject"));
            }
            if AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
                return Err(last_error("AssignProcessToJobObject"));
            }
            let _ = JOB.set(job as usize);
        }
        Ok(())
    }

    /// Peak committed memory the job saw: (peak of any one process, peak of the job), if a cap was applied.
    pub fn job_peak_memory() -> Option<(u64, u64)> {
        let job = *JOB.get()? as Handle;
        let mut ext = JobExtendedLimit::default();
        // SAFETY: the job handle is ours and the struct is sized.
        let ok = unsafe { QueryInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, &mut ext as *mut _ as *mut c_void, std::mem::size_of::<JobExtendedLimit>() as u32, std::ptr::null_mut()) };
        if ok == 0 {
            return None;
        }
        Some((ext.peak_process_memory_used as u64, ext.peak_job_memory_used as u64))
    }
}

// ================================================================================================ Linux (untested)
#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::ffi::{c_int, c_void, CString};
    use std::os::unix::ffi::OsStrExt;

    const O_RDONLY: c_int = 0;
    const O_DIRECT: c_int = 0o40000;
    const O_CLOEXEC: c_int = 0o2000000;
    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    const MAP_PRIVATE: c_int = 2;
    const MAP_ANONYMOUS: c_int = 0x20;
    const MADV_HUGEPAGE: c_int = 14;

    extern "C" {
        fn open(path: *const i8, flags: c_int, ...) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn pread(fd: c_int, buf: *mut c_void, n: usize, off: i64) -> isize;
        fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
        fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }

    pub fn try_enable_lock_memory_privilege() -> Result<(), String> {
        Ok(())
    }

    /// Transparent huge pages: 2 MiB when /sys says so, else 0.
    pub fn large_page_minimum() -> usize {
        match std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
            Ok(s) if !s.contains("[never]") => 2 << 20,
            _ => 0,
        }
    }

    impl AlignedBuf {
        pub fn new(len: usize, want_large: bool) -> io::Result<AlignedBuf> {
            let len = align_up(len.max(4096) as u64, 4096) as usize;
            // SAFETY: anonymous private mapping; MAP_FAILED is checked.
            let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
            if p as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            let mut large = false;
            if want_large {
                if large_page_minimum() == 0 {
                    note_large_page_fallback("transparent hugepages are disabled (/sys/kernel/mm/transparent_hugepage/enabled)".into());
                } else {
                    // SAFETY: advising our own mapping; best effort (the kernel may still back it with 4 KiB pages).
                    large = unsafe { madvise(p, len, MADV_HUGEPAGE) } == 0;
                    if !large {
                        note_large_page_fallback(format!("madvise(MADV_HUGEPAGE) failed: {}", io::Error::last_os_error()));
                    }
                }
            }
            Ok(AlignedBuf { ptr: p as *mut u8, len, large_pages: large })
        }
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            // SAFETY: mapping we created, unmapped once.
            unsafe { munmap(self.ptr as *mut c_void, self.len) };
        }
    }

    pub struct DirectFile {
        fd: c_int,
        pub sector: usize,
        pub chunk: usize,
        pub qd: usize,
        pub file_size: u64,
    }

    /// The logical block size of the device holding `path`, from sysfs; 4096 when it cannot be found.
    fn device_block_size(path: &Path) -> usize {
        use std::os::unix::fs::MetadataExt;
        let dev = match std::fs::metadata(path) {
            Ok(m) => m.dev(),
            Err(_) => return 4096,
        };
        let (major, minor) = ((dev >> 8) & 0xfff, (dev & 0xff) | ((dev >> 12) & 0xfff00));
        for rel in ["queue/logical_block_size", "../queue/logical_block_size"] {
            if let Ok(s) = std::fs::read_to_string(format!("/sys/dev/block/{major}:{minor}/{rel}")) {
                if let Ok(v) = s.trim().parse::<usize>() {
                    return v;
                }
            }
        }
        4096
    }

    impl DirectFile {
        pub fn open(path: &Path, chunk: usize, qd: usize) -> io::Result<DirectFile> {
            let c = CString::new(path.as_os_str().as_bytes()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path"))?;
            // SAFETY: documented libc open.
            let fd = unsafe { open(c.as_ptr() as *const i8, O_RDONLY | O_DIRECT | O_CLOEXEC) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let sector = device_block_size(path).max(4096);
            let file_size = std::fs::metadata(path)?.len();
            let chunk = align_up(chunk.max(sector) as u64, sector as u64) as usize;
            // qd > 1 needs io_uring / aio; reads are issued one chunk at a time here (untested path).
            Ok(DirectFile { fd, sector, chunk, qd: qd.max(1).min(1), file_size })
        }

        pub fn read_at(&mut self, offset: u64, buf: &AlignedBuf, len: usize) -> io::Result<usize> {
            assert!(len <= buf.len(), "DirectFile::read_at: {len} bytes into a {}-byte buffer", buf.len());
            let buf = buf.as_mut_ptr();
            assert!(offset.is_multiple_of(self.sector as u64) && len.is_multiple_of(self.sector) && (buf as usize).is_multiple_of(self.sector), "DirectFile::read_at: unaligned");
            let mut done = 0usize;
            while done < len {
                let this = (len - done).min(self.chunk);
                // SAFETY: the range is inside the caller's buffer.
                let r = unsafe { pread(self.fd, buf.add(done) as *mut c_void, this, (offset + done as u64) as i64) };
                if r < 0 {
                    let e = io::Error::last_os_error();
                    if e.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(e);
                }
                if r == 0 {
                    break; // EOF
                }
                done += r as usize;
                if (r as usize) < this {
                    break;
                }
            }
            Ok(done)
        }
    }

    impl Drop for DirectFile {
        fn drop(&mut self) {
            // SAFETY: fd we opened.
            unsafe { close(self.fd) };
        }
    }

    pub fn mem_status() -> Option<(u64, u64)> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb = |key: &str| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).map(|v| v * 1024);
        Some((kb("MemTotal:")?, kb("MemAvailable:")?))
    }

    /// The cap on Linux is applied from outside (`systemd-run -p MemoryMax=`, `scripts/ladder.sh`).
    pub fn apply_job_memory_limit(_bytes: u64) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "no in-process memory cap on Linux; use systemd-run -p MemoryMax= (scripts/ladder.sh)"))
    }

    pub fn job_peak_memory() -> Option<(u64, u64)> {
        None
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
compile_error!("aqueduct-core::os: only Windows and Linux are supported");

pub use imp::{apply_job_memory_limit, job_peak_memory, large_page_minimum, mem_status, try_enable_lock_memory_privilege, DirectFile};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_arithmetic() {
        assert_eq!(align_down(4097, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(8192, 4096), 8192);
        // every start offset modulo the sector fits: superset length <= arena_bytes
        for span in [1u64, 4095, 4096, 4097, 269_681_536, 254_568_448] {
            for start in [0u64, 1, 32, 896, 4095, 4096, 12345] {
                let (a, b) = (align_down(start, 4096), align_up(start + span, 4096));
                assert!(b - a <= arena_bytes(span, 4096), "span {span} start {start}");
            }
        }
    }

    #[test]
    fn aligned_buf_is_sector_aligned_and_writable() {
        let b = AlignedBuf::new(1 << 20, false).unwrap();
        assert_eq!(b.as_ptr() as usize % 4096, 0);
        assert!(b.len() >= 1 << 20);
        // SAFETY: our own buffer, nobody else touches it.
        unsafe {
            std::ptr::write_bytes(b.as_mut_ptr(), 0xAB, b.len());
            assert_eq!(b.slice(4096, 8)[3], 0xAB);
        }
    }

    #[test]
    fn direct_read_matches_buffered_read() {
        // read a sector-aligned window of this crate's own Cargo.lock / a source file through both paths
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("gguf.rs");
        let whole = std::fs::read(&path).unwrap();
        let mut f = DirectFile::open(&path, 1 << 16, 2).unwrap();
        let sector = f.sector as u64;
        assert!(sector >= 512 && sector.is_power_of_two(), "sector {sector}");
        let want = align_up(whole.len() as u64, sector) as usize;
        let buf = AlignedBuf::new(want, false).unwrap();
        let got = f.read_at(0, &buf, want).unwrap();
        assert_eq!(got, whole.len(), "bytes transferred up to EOF");
        // SAFETY: the read has completed and nothing else writes the buffer.
        assert_eq!(unsafe { buf.slice(0, whole.len()) }, &whole[..]);
        println!("sector {sector}, file {} bytes, {got} transferred", whole.len());
    }

    #[test]
    fn memory_status_is_sane() {
        let (total, avail) = mem_status().expect("mem_status");
        assert!(total > 0 && avail <= total, "total {total} avail {avail}");
    }
}
