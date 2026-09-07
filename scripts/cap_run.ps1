# scripts/cap_run.ps1 -- Phase 6.1: run any program under a Windows job object memory cap, from PowerShell.
#
#   powershell -ExecutionPolicy Bypass -Command "& scripts\cap_run.ps1 -Cap 5G [-Out out.txt] [-Err err.txt]
#       [-Report report.json] [-NoWait] -Exe <program> [-Arguments arg1,arg2,...]"
# (the child's arguments are one string array: PowerShell's binder would otherwise read the child's own
# switches as this script's)
#
# The Phase 4 ladder capped aqueduct by having the engine put ITSELF into a job object (--job-limit) before it
# allocated, which the Phase 4 report lists under "did NOT do: enforce the cap from PowerShell with
# CreateProcess-suspended". This script is that launcher, for programs that cannot cap themselves (llama.cpp):
#
#   1. CreateJobObject; SetInformationJobObject with JOB_OBJECT_LIMIT_JOB_MEMORY and
#      JOB_OBJECT_LIMIT_PROCESS_MEMORY = cap (committed memory: what aqueduct's --job-limit enforces) and
#      JOB_OBJECT_LIMIT_WORKINGSET with the maximum working set = cap.
#   2. CreateProcess with CREATE_SUSPENDED, so not one byte is allocated before the cap applies.
#   3. AssignProcessToJobObject; SetProcessWorkingSetSizeEx(max = cap, QUOTA_LIMITS_HARDWS_MAX_ENABLE) so the
#      working-set maximum is a hard limit (a soft maximum is only honoured under memory pressure). Both
#      working-set calls need SeIncreaseWorkingSetPrivilege, held by every user but disabled in the token
#      until AdjustTokenPrivileges enables it. If the JOB-level working-set flag is still refused (error 1314:
#      it wants an elevated token on this machine), the job keeps the commit limits and the per-process hard
#      maximum alone bounds the working set; the report says which ("working-set limit on: job+process" or
#      "process").
#   4. ResumeThread; wait; read the peaks: the job's PeakJobMemoryUsed / PeakProcessMemoryUsed (commit),
#      GetProcessMemoryInfo's PeakWorkingSetSize and the working set sampled every 250 ms, the page-fault
#      count, GetProcessIoCounters' ReadTransferCount (ReadFile bytes; page-ins of a mapped file are NOT
#      counted there), and the physical disk's cumulative read bytes on the model's volume before and after
#      (Win32_PerfRawData_PerfDisk_PhysicalDisk, a raw cumulative counter, so it does not suffer from the
#      rate-counter quirk on this machine).
#
# WHY BOTH LIMITS. A commit cap alone is blind to a memory-mapped model file: mapped file pages are not
# committed memory, so llama.cpp (mmap by default) would run at any commit cap with the whole 17.8 GB file
# resident in the page cache. The working-set cap is what bounds a mapped file's resident pages. What the
# working-set cap does NOT do is take the pages out of RAM: on a 32 GB machine the trimmed pages stay in the
# OS standby list and come back by soft fault, without a disk read. The disk-bytes column says which happened.
#
# -NoWait resumes the child, prints its PID and returns without waiting; the job (and its limits) persists
# while the child lives, because JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is deliberately not set. The caller reads
# the child's peaks itself (Get-Process: PeakWorkingSet64, PeakPagedMemorySize64).
#
# Exit code: the child's (or 0 with -NoWait). The one-line summary and the JSON report carry the peaks and
# `under` = every peak <= cap (the working set may overshoot a hard maximum by a few pages before the trim, so
# it is allowed 64 MiB; the commit peaks are exact).
param(
    [Parameter(Mandatory = $true)][string]$Cap,
    [string]$Out = "",
    [string]$Err = "",
    [string]$Report = "",
    [string]$DiskVolume = "C:",
    [switch]$NoWait,
    [Parameter(Mandatory = $true)][string]$Exe,
    [string[]]$Arguments = @()
)
$ErrorActionPreference = "Stop"
function Parse-Size([string]$s) {
    if ($s -match '^([0-9.]+)G$') { return [long]([double]$Matches[1] * 1073741824) }
    if ($s -match '^([0-9.]+)M$') { return [long]([double]$Matches[1] * 1048576) }
    return [long]$s
}
$capBytes = Parse-Size $Cap
$exe = $Exe
$resolved = Get-Command $exe -ErrorAction SilentlyContinue
if ($resolved) { $exe = $resolved.Source } else { $exe = (Resolve-Path $exe).Path }
$args2 = @($Arguments)
function Quote-Arg([string]$a) { if ($a -match '[\s"]' -or $a -eq "") { return '"' + ($a -replace '"', '\"') + '"' } else { return $a } }
$cmdline = (@($exe) + $args2 | ForEach-Object { Quote-Arg $_ }) -join " "
$tmp = Join-Path $env:TEMP "cap_run"
New-Item -ItemType Directory -Force $tmp | Out-Null
$echoOut = $false
$stamp = "{0}_{1}" -f $PID, (Get-Date -Format "HHmmssfff")
if ($Out -eq "") { $Out = Join-Path $tmp "out_$stamp.txt"; $echoOut = $true }
if ($Err -eq "") { $Err = Join-Path $tmp "err_$stamp.txt"; $echoOut = $true }
$Out = [System.IO.Path]::GetFullPath($Out); $Err = [System.IO.Path]::GetFullPath($Err)

if (-not ("CapRun.Native" -as [type])) {
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;
namespace CapRun {
public static class Native {
    [StructLayout(LayoutKind.Sequential)] public struct IO_COUNTERS { public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount; }
    [StructLayout(LayoutKind.Sequential)] public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize; public uint ActiveProcessLimit;
        public UIntPtr Affinity; public uint PriorityClass; public uint SchedulingClass; }
    [StructLayout(LayoutKind.Sequential)] public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit; public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] public struct STARTUPINFO {
        public uint cb; public string lpReserved; public string lpDesktop; public string lpTitle;
        public uint dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        public ushort wShowWindow, cbReserved2; public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError; }
    [StructLayout(LayoutKind.Sequential)] public struct PROCESS_INFORMATION { public IntPtr hProcess, hThread; public uint dwProcessId, dwThreadId; }
    [StructLayout(LayoutKind.Sequential)] public struct SECURITY_ATTRIBUTES { public uint nLength; public IntPtr lpSecurityDescriptor; public int bInheritHandle; }
    [StructLayout(LayoutKind.Sequential)] public struct PROCESS_MEMORY_COUNTERS {
        public uint cb, PageFaultCount; public UIntPtr PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
        QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage; }
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] public static extern IntPtr CreateJobObjectW(IntPtr attrs, string name);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, int len);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool QueryInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, int len, IntPtr ret);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] public static extern bool CreateProcessW(string app, StringBuilder cmd, IntPtr pa, IntPtr ta, bool inherit, uint flags, IntPtr env, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern uint WaitForSingleObject(IntPtr h, uint ms);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool GetExitCodeProcess(IntPtr h, out uint code);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] public static extern IntPtr CreateFileW(string path, uint access, uint share, ref SECURITY_ATTRIBUTES sa, uint disp, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr GetStdHandle(int n);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool K32GetProcessMemoryInfo(IntPtr h, ref PROCESS_MEMORY_COUNTERS c, uint cb);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool GetProcessIoCounters(IntPtr h, out IO_COUNTERS c);
    public const uint JOB_OBJECT_LIMIT_WORKINGSET = 0x1, JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100, JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200;
    public const int JobObjectExtendedLimitInformation = 9;
    public const uint CREATE_SUSPENDED = 0x4, STARTF_USESTDHANDLES = 0x100;
    public const uint QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x2, QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x4;
    [StructLayout(LayoutKind.Sequential)] public struct LUID { public uint LowPart; public int HighPart; }
    [StructLayout(LayoutKind.Sequential)] public struct TOKEN_PRIVILEGES { public uint PrivilegeCount; public LUID Luid; public uint Attributes; }
    [DllImport("advapi32.dll", SetLastError = true)] public static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)] public static extern bool LookupPrivilegeValueW(string system, string name, out LUID luid);
    [DllImport("advapi32.dll", SetLastError = true)] public static extern bool AdjustTokenPrivileges(IntPtr token, bool disableAll, ref TOKEN_PRIVILEGES newState, uint len, IntPtr prev, IntPtr ret);
    [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
    // a job working-set limit needs SeIncreaseWorkingSetPrivilege, which every user holds but which is disabled in the token until enabled
    public static string EnablePrivilege(string name) {
        IntPtr tok; if (!OpenProcessToken(GetCurrentProcess(), 0x20 | 0x8, out tok)) return "OpenProcessToken: " + Marshal.GetLastWin32Error();
        var tp = new TOKEN_PRIVILEGES(); tp.PrivilegeCount = 1; tp.Attributes = 0x2;
        if (!LookupPrivilegeValueW(null, name, out tp.Luid)) return "LookupPrivilegeValue(" + name + "): " + Marshal.GetLastWin32Error();
        if (!AdjustTokenPrivileges(tok, false, ref tp, (uint)Marshal.SizeOf(tp), IntPtr.Zero, IntPtr.Zero)) return "AdjustTokenPrivileges: " + Marshal.GetLastWin32Error();
        int e = Marshal.GetLastWin32Error(); CloseHandle(tok);
        if (e == 1300) return "AdjustTokenPrivileges: " + name + " is not held by this user (ERROR_NOT_ALL_ASSIGNED)";
        return "";
    }
    public static IntPtr Job; public static PROCESS_INFORMATION Pi; public static string WsMechanism = "";
    public static string Launch(string cmdline, string outPath, string errPath, ulong cap, ulong minWs) {
        string pe = EnablePrivilege("SeIncreaseWorkingSetPrivilege"); if (pe != "") return pe;
        Job = CreateJobObjectW(IntPtr.Zero, null);
        if (Job == IntPtr.Zero) return "CreateJobObjectW: " + Marshal.GetLastWin32Error();
        var ext = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        ext.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_WORKINGSET;
        ext.BasicLimitInformation.MinimumWorkingSetSize = (UIntPtr)minWs;
        ext.BasicLimitInformation.MaximumWorkingSetSize = (UIntPtr)cap;
        ext.ProcessMemoryLimit = (UIntPtr)cap; ext.JobMemoryLimit = (UIntPtr)cap;
        if (!SetInformationJobObject(Job, JobObjectExtendedLimitInformation, ref ext, Marshal.SizeOf(ext))) {
            int e1 = Marshal.GetLastWin32Error();
            if (e1 != 1314) return "SetInformationJobObject: " + e1;
            // ERROR_PRIVILEGE_NOT_HELD: the job-level working-set limit is refused to this token; keep the commit
            // limits on the job and rely on the per-process hard working-set maximum set below
            ext.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            ext.BasicLimitInformation.MinimumWorkingSetSize = UIntPtr.Zero; ext.BasicLimitInformation.MaximumWorkingSetSize = UIntPtr.Zero;
            if (!SetInformationJobObject(Job, JobObjectExtendedLimitInformation, ref ext, Marshal.SizeOf(ext))) return "SetInformationJobObject (commit only): " + Marshal.GetLastWin32Error();
            WsMechanism = "process";
        } else { WsMechanism = "job+process"; }
        var sa = new SECURITY_ATTRIBUTES(); sa.nLength = (uint)Marshal.SizeOf(sa); sa.bInheritHandle = 1;
        IntPtr hOut = CreateFileW(outPath, 0x40000000, 0x3, ref sa, 2, 0x80, IntPtr.Zero);
        if (hOut == (IntPtr)(-1)) return "CreateFileW(out): " + Marshal.GetLastWin32Error();
        IntPtr hErr = CreateFileW(errPath, 0x40000000, 0x3, ref sa, 2, 0x80, IntPtr.Zero);
        if (hErr == (IntPtr)(-1)) return "CreateFileW(err): " + Marshal.GetLastWin32Error();
        var si = new STARTUPINFO(); si.cb = (uint)Marshal.SizeOf(si); si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(-10); si.hStdOutput = hOut; si.hStdError = hErr;
        var sb = new StringBuilder(cmdline);
        if (!CreateProcessW(null, sb, IntPtr.Zero, IntPtr.Zero, true, CREATE_SUSPENDED, IntPtr.Zero, null, ref si, out Pi)) return "CreateProcessW: " + Marshal.GetLastWin32Error();
        CloseHandle(hOut); CloseHandle(hErr);
        if (!AssignProcessToJobObject(Job, Pi.hProcess)) return "AssignProcessToJobObject: " + Marshal.GetLastWin32Error();
        if (!SetProcessWorkingSetSizeEx(Pi.hProcess, (UIntPtr)minWs, (UIntPtr)cap, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)) return "SetProcessWorkingSetSizeEx: " + Marshal.GetLastWin32Error();
        if (ResumeThread(Pi.hThread) == 0xFFFFFFFF) return "ResumeThread: " + Marshal.GetLastWin32Error();
        return "";
    }
    public static bool Wait(uint ms) { return WaitForSingleObject(Pi.hProcess, ms) == 0; }
    public static uint ExitCode() { uint c; GetExitCodeProcess(Pi.hProcess, out c); return c; }
    public static PROCESS_MEMORY_COUNTERS Mem() { var c = new PROCESS_MEMORY_COUNTERS(); c.cb = (uint)Marshal.SizeOf(c); K32GetProcessMemoryInfo(Pi.hProcess, ref c, c.cb); return c; }
    public static JOBOBJECT_EXTENDED_LIMIT_INFORMATION JobInfo() { var ext = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION(); QueryInformationJobObject(Job, JobObjectExtendedLimitInformation, ref ext, Marshal.SizeOf(ext), IntPtr.Zero); return ext; }
    public static IO_COUNTERS Io() { IO_COUNTERS c; GetProcessIoCounters(Pi.hProcess, out c); return c; }
    public static void Close() { CloseHandle(Pi.hThread); CloseHandle(Pi.hProcess); CloseHandle(Job); }
}
}
"@
}

function Disk-ReadBytes([string]$vol) {
    try {
        $d = Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -ErrorAction Stop | Where-Object { $_.Name -like "*$vol*" } | Select-Object -First 1
        if ($d) { return [long]$d.DiskReadBytesPersec }
    } catch {}
    return $null
}

$diskBefore = Disk-ReadBytes $DiskVolume
$t0 = Get-Date
$e = [CapRun.Native]::Launch($cmdline, $Out, $Err, [uint64]$capBytes, [uint64](64 * 1048576))
if ($e -ne "") { throw "cap_run: $e" }
$childPid = [CapRun.Native]::Pi.dwProcessId
$wsMech = [CapRun.Native]::WsMechanism
if ($NoWait) {
    Write-Host ("cap_run: pid {0} running under cap {1} ({2:N0} bytes: job commit limits; hard working-set maximum on {3}); not waiting" -f $childPid, $Cap, $capBytes, $wsMech)
    if ($Report -ne "") { [pscustomobject]@{ pid = $childPid; cap = $capBytes; command = $cmdline; out = $Out; err = $Err; started = $t0.ToString("o"); working_set_limit = $wsMech } | ConvertTo-Json | Set-Content $Report }
    Write-Output $childPid
    exit 0
}
$sampledPeak = [long]0
$faults = 0
while (-not [CapRun.Native]::Wait(250)) {
    $m = [CapRun.Native]::Mem()
    $ws = [long]$m.WorkingSetSize.ToUInt64(); if ($ws -gt $sampledPeak) { $sampledPeak = $ws }
}
$wall = ((Get-Date) - $t0).TotalSeconds
$code = [CapRun.Native]::ExitCode()
$m = [CapRun.Native]::Mem()
$j = [CapRun.Native]::JobInfo()
$io = [CapRun.Native]::Io()
[CapRun.Native]::Close()
$diskAfter = Disk-ReadBytes $DiskVolume
$diskDelta = $null
if ($diskBefore -ne $null -and $diskAfter -ne $null) { $diskDelta = $diskAfter - $diskBefore }
$peakWs = [Math]::Max([long]$m.PeakWorkingSetSize.ToUInt64(), $sampledPeak)
$peakCommit = [long]$m.PeakPagefileUsage.ToUInt64()
$jobPeak = [long]$j.PeakJobMemoryUsed.ToUInt64()
# the hard working-set maximum lets the working set overshoot by a few pages before the trim; the commit limits are exact
$tol = 64 * 1048576
$under = ($peakWs -le $capBytes + $tol) -and ($peakCommit -le $capBytes) -and ($jobPeak -le $capBytes)
$rep = [pscustomobject]@{
    command = $cmdline; cap = $capBytes; exit_code = [int]$code; wall_s = [Math]::Round($wall, 3)
    peak_working_set = $peakWs; peak_working_set_sampled = $sampledPeak; peak_commit = $peakCommit
    job_peak_process = [long]$j.PeakProcessMemoryUsed.ToUInt64(); job_peak_job = $jobPeak; page_faults = [long]$m.PageFaultCount
    io_read_bytes = [long]$io.ReadTransferCount; disk_read_bytes = $diskDelta; disk_volume = $DiskVolume; under = $under
    working_set_limit = $wsMech; out = $Out; err = $Err
}
if ($Report -ne "") { $rep | ConvertTo-Json | Set-Content $Report }
Write-Host ("cap_run: exit {0} after {1:F1} s under cap {2} (working-set limit on: $wsMech); peak working set {3:F3} GB (sampled {4:F3}), peak commit {5:F3} GB, job peak {6:F3} GB, page faults {7:N0}, ReadFile bytes {8:F3} GB, {9} disk read during the run {10}; under cap: {11}" -f $code, $wall, $Cap, ($peakWs / 1e9), ($sampledPeak / 1e9), ($peakCommit / 1e9), ($jobPeak / 1e9), $rep.page_faults, ($rep.io_read_bytes / 1e9), $DiskVolume, $(if ($diskDelta -ne $null) { "{0:F3} GB" -f ($diskDelta / 1e9) } else { "n/a" }), $(if ($under) { "yes" } else { "NO" }))
if ($echoOut) {
    if (Test-Path $Out) { Get-Content $Out | ForEach-Object { Write-Host $_ } }
    if (Test-Path $Err) { Get-Content $Err | ForEach-Object { Write-Host $_ } }
}
exit [int]$code
