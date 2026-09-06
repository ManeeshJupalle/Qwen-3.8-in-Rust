//! What kind of drive holds a path (`aqueduct doctor`, Phase 6): the bus it sits on and whether it has a
//! seek penalty, asked of the OS rather than inferred from a speed. Windows: `IOCTL_STORAGE_QUERY_PROPERTY`
//! on the volume (`StorageDeviceProperty` for the bus and the product name, `StorageDeviceSeekPenaltyProperty`
//! for the spindle). Linux: the block device behind the path's `st_dev` under `/sys/dev/block`, walked up
//! through partitions and device-mapper slaves to the disk, then `queue/rotational` and the device's sysfs
//! path for the bus. The Linux path compiles and has not been run (as the rest of the Linux port).

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskClass {
    /// NVMe bus, no seek penalty.
    Nvme,
    /// A solid-state drive on any other bus (SATA, USB, SCSI, RAID, SD, ...).
    Ssd,
    /// The OS reports a seek penalty: a spinning disk.
    Hdd,
    /// The query failed; nothing is assumed.
    Unknown,
}

#[derive(Debug, Clone)]
pub struct DiskInfo {
    /// The volume or device the path resolves to (`C:`, `/dev/nvme0n1`).
    pub volume: String,
    /// Bus name as the OS reports it (`NVMe`, `SATA`, `USB`, ...).
    pub bus: String,
    /// Product name when the OS gives one.
    pub product: Option<String>,
    /// Seek penalty (spinning media), when the OS answers.
    pub seek_penalty: Option<bool>,
    pub class: DiskClass,
    /// Why the query fell short, if it did.
    pub note: Option<String>,
}

impl DiskInfo {
    pub fn describe(&self) -> String {
        let mut s = format!("{} {}", self.volume, self.bus);
        if let Some(p) = &self.product {
            s.push_str(&format!(" ({p})"));
        }
        match self.seek_penalty {
            Some(true) => s.push_str(", seek penalty: yes (spinning disk)"),
            Some(false) => s.push_str(", seek penalty: no"),
            None => s.push_str(", seek penalty: unknown"),
        }
        if let Some(n) = &self.note {
            s.push_str(&format!("; {n}"));
        }
        s
    }
}

/// The nearest existing ancestor of `path` (the path itself when it exists).
pub fn nearest_existing(path: &Path) -> PathBuf {
    let mut p = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir().map(|d| d.join(path)).unwrap_or_else(|_| path.to_path_buf()) };
    loop {
        if p.exists() {
            return p;
        }
        match p.parent() {
            Some(parent) if parent != p => p = parent.to_path_buf(),
            _ => return p,
        }
    }
}

fn classify(bus: &str, seek_penalty: Option<bool>) -> DiskClass {
    match (seek_penalty, bus) {
        (Some(true), _) => DiskClass::Hdd,
        (_, "NVMe") => DiskClass::Nvme,
        (Some(false), _) => DiskClass::Ssd,
        (None, "unknown") => DiskClass::Unknown,
        (None, _) => DiskClass::Ssd,
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
    const FILE_SHARE_READ: u32 = 1;
    const FILE_SHARE_WRITE: u32 = 2;
    const OPEN_EXISTING: u32 = 3;
    const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D_1400;
    const STORAGE_DEVICE_PROPERTY: u32 = 0;
    const STORAGE_DEVICE_SEEK_PENALTY_PROPERTY: u32 = 7;
    const PROPERTY_STANDARD_QUERY: u32 = 0;

    #[repr(C)]
    struct StoragePropertyQuery {
        property_id: u32,
        query_type: u32,
        additional: [u8; 1],
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(name: *const u16, access: u32, share: u32, sec: *mut c_void, disp: u32, flags: u32, template: Handle) -> Handle;
        fn DeviceIoControl(h: Handle, code: u32, inp: *mut c_void, in_len: u32, out: *mut c_void, out_len: u32, returned: *mut u32, ov: *mut c_void) -> i32;
        fn CloseHandle(h: Handle) -> i32;
        fn GetLastError() -> u32;
    }

    fn bus_name(code: u32) -> String {
        match code {
            0 => "unknown".into(),
            1 => "SCSI".into(),
            2 => "ATAPI".into(),
            3 => "ATA".into(),
            4 => "IEEE 1394".into(),
            5 => "SSA".into(),
            6 => "Fibre Channel".into(),
            7 => "USB".into(),
            8 => "RAID".into(),
            9 => "iSCSI".into(),
            10 => "SAS".into(),
            11 => "SATA".into(),
            12 => "SD".into(),
            13 => "MMC".into(),
            14 => "virtual".into(),
            15 => "file-backed virtual".into(),
            16 => "Storage Spaces".into(),
            17 => "NVMe".into(),
            18 => "SCM".into(),
            19 => "UFS".into(),
            n => format!("bus type {n}"),
        }
    }

    fn query(h: Handle, property: u32, out: &mut [u8]) -> Result<usize, String> {
        let mut q = StoragePropertyQuery { property_id: property, query_type: PROPERTY_STANDARD_QUERY, additional: [0] };
        let mut returned: u32 = 0;
        // SAFETY: documented kernel32 call; the query and output buffers outlive it and their sizes are passed.
        let ok = unsafe { DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &mut q as *mut _ as *mut c_void, std::mem::size_of::<StoragePropertyQuery>() as u32, out.as_mut_ptr() as *mut c_void, out.len() as u32, &mut returned, std::ptr::null_mut()) };
        if ok == 0 {
            // SAFETY: plain kernel32 call.
            return Err(format!("IOCTL_STORAGE_QUERY_PROPERTY({property}) failed: os error {}", unsafe { GetLastError() }));
        }
        Ok(returned as usize)
    }

    fn u32_at(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }

    fn cstr_at(b: &[u8], off: usize) -> Option<String> {
        if off == 0 || off >= b.len() {
            return None;
        }
        let end = b[off..].iter().position(|&c| c == 0).map(|n| off + n).unwrap_or(b.len());
        let s = String::from_utf8_lossy(&b[off..end]).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    pub fn disk_info(path: &Path) -> DiskInfo {
        let existing = nearest_existing(path);
        let canon = std::fs::canonicalize(&existing).unwrap_or(existing);
        let s = canon.to_string_lossy().to_string();
        let s = s.strip_prefix(r"\\?\").unwrap_or(&s).to_string();
        let letter = s.chars().next().filter(|c| c.is_ascii_alphabetic() && s.chars().nth(1) == Some(':'));
        let Some(letter) = letter else {
            return DiskInfo { volume: s.chars().take(2).collect(), bus: "unknown".into(), product: None, seek_penalty: None, class: DiskClass::Unknown, note: Some("not a drive-letter path (a UNC or unresolved path); the bus query needs a volume".into()) };
        };
        let volume = format!("{letter}:");
        let name: Vec<u16> = std::ffi::OsStr::new(&format!(r"\\.\{volume}")).encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: documented kernel32 call; access 0 is enough for the storage property query.
        let h = unsafe { CreateFileW(name.as_ptr(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null_mut(), OPEN_EXISTING, 0, std::ptr::null_mut()) };
        if h == INVALID_HANDLE {
            // SAFETY: plain kernel32 call.
            let e = unsafe { GetLastError() };
            return DiskInfo { volume, bus: "unknown".into(), product: None, seek_penalty: None, class: DiskClass::Unknown, note: Some(format!("open \\\\.\\{letter}: failed: os error {e}")) };
        }
        let mut buf = vec![0u8; 4096];
        let (bus, product, mut note) = match query(h, STORAGE_DEVICE_PROPERTY, &mut buf) {
            Ok(n) if n >= 32 => {
                let bus = bus_name(u32_at(&buf, 28));
                let vendor = cstr_at(&buf, u32_at(&buf, 12) as usize);
                let product = cstr_at(&buf, u32_at(&buf, 16) as usize);
                let product = match (vendor, product) {
                    (Some(v), Some(p)) => Some(format!("{v} {p}")),
                    (None, Some(p)) => Some(p),
                    (Some(v), None) => Some(v),
                    (None, None) => None,
                };
                (bus, product, None)
            }
            Ok(n) => ("unknown".to_string(), None, Some(format!("device descriptor too short ({n} bytes)"))),
            Err(e) => ("unknown".to_string(), None, Some(e)),
        };
        let seek_penalty = match query(h, STORAGE_DEVICE_SEEK_PENALTY_PROPERTY, &mut buf) {
            Ok(n) if n >= 9 => Some(buf[8] != 0),
            Ok(n) => {
                note.get_or_insert_with(String::new).push_str(&format!(" seek-penalty descriptor too short ({n} bytes)"));
                None
            }
            Err(e) => {
                let n = note.get_or_insert_with(String::new);
                if !n.is_empty() {
                    n.push_str("; ");
                }
                n.push_str(&e);
                None
            }
        };
        // SAFETY: closing the handle we opened.
        unsafe { CloseHandle(h) };
        let class = classify(&bus, seek_penalty);
        DiskInfo { volume, bus, product, seek_penalty, class, note }
    }
}

// ================================================================================================ Linux
#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    fn dev_major_minor(dev: u64) -> (u64, u64) {
        let major = ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff);
        let minor = ((dev >> 12) & 0xffff_ff00) | (dev & 0xff);
        (major, minor)
    }

    /// From a block device's sysfs directory to the whole disk it lives on: up out of a partition, down
    /// through device-mapper / md slaves.
    fn to_disk(mut dir: PathBuf) -> PathBuf {
        for _ in 0..8 {
            if dir.join("partition").exists() {
                if let Some(p) = dir.parent() {
                    dir = p.to_path_buf();
                    continue;
                }
            }
            let slaves = dir.join("slaves");
            if let Ok(rd) = std::fs::read_dir(&slaves) {
                if let Some(first) = rd.flatten().next() {
                    if let Ok(target) = std::fs::canonicalize(first.path()) {
                        dir = target;
                        continue;
                    }
                }
            }
            break;
        }
        dir
    }

    pub fn disk_info(path: &Path) -> DiskInfo {
        let existing = nearest_existing(path);
        let unknown = |volume: String, note: String| DiskInfo { volume, bus: "unknown".into(), product: None, seek_penalty: None, class: DiskClass::Unknown, note: Some(note) };
        let meta = match std::fs::metadata(&existing) {
            Ok(m) => m,
            Err(e) => return unknown(existing.display().to_string(), format!("stat: {e}")),
        };
        let (major, minor) = dev_major_minor(meta.dev());
        let link = PathBuf::from(format!("/sys/dev/block/{major}:{minor}"));
        let dir = match std::fs::canonicalize(&link) {
            Ok(d) => d,
            Err(e) => return unknown(format!("{major}:{minor}"), format!("{}: {e}", link.display())),
        };
        let disk = to_disk(dir);
        let name = disk.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| format!("{major}:{minor}"));
        let volume = format!("/dev/{name}");
        let rotational = std::fs::read_to_string(disk.join("queue/rotational")).ok().map(|s| s.trim() == "1");
        let product = std::fs::read_to_string(disk.join("device/model")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let full = disk.to_string_lossy().to_string();
        let bus = if name.starts_with("nvme") || full.contains("/nvme/") {
            "NVMe"
        } else if name.starts_with("mmcblk") {
            "SD/MMC"
        } else if full.contains("/usb") {
            "USB"
        } else if full.contains("/ata") {
            "SATA"
        } else if full.contains("/virtio") || name.starts_with("vd") || name.starts_with("xvd") {
            "virtual"
        } else if name.starts_with("sd") {
            "SCSI"
        } else {
            "unknown"
        }
        .to_string();
        let class = classify(&bus, rotational);
        DiskInfo { volume, bus, product, seek_penalty: rotational, class, note: Some("Linux: from sysfs (this path has not been run on Linux)".into()) }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    use super::*;
    pub fn disk_info(path: &Path) -> DiskInfo {
        DiskInfo { volume: nearest_existing(path).display().to_string(), bus: "unknown".into(), product: None, seek_penalty: None, class: DiskClass::Unknown, note: Some("no disk query on this platform".into()) }
    }
}

pub use imp::disk_info;
