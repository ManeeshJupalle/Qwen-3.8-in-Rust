"""Summarise a `tools/bench_rounds.ps1` log: per round, the membw ceiling of that round (best of 6 or 12
threads) and every kernel line at 6 and 12 threads as GB/s and as % of that ceiling, split by binary
(the `-Exe` under test and the `-Baseline`) and by matrix shape; then the range over rounds per kernel.

Usage: python tools/bench_rounds_summary.py docs/data/bench_rounds.log [--threads 6,12]
"""
import argparse
import re
from collections import defaultdict

RUN_RE = re.compile(r"^# > (\S+) bench (membw|kernels)(.*)$")
MEMBW_RE = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+")
KERNEL_RE = re.compile(r"^(Q\w+)\s+(q8_\w)\s+(avx2|scalar)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
STATE_RE = re.compile(r"^# state \[(.*?)\] (\S+): (.*)$")
ROUND_RE = re.compile(r"^# ==== round (\d+) of (\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--threads", default="6,12")
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    rounds = []  # list of dicts: {"membw": {thr: best}, "kernels": {(exe, rows): [(kernel, act, thr, gbs)]}, "states": []}
    cur = None
    exe = None
    kind = None
    rows = None
    exes = []
    for line in open(a.log, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = ROUND_RE.match(line)
        if m:
            cur = {"n": int(m.group(1)), "membw": {}, "kernels": defaultdict(list), "states": []}
            rounds.append(cur)
            continue
        m = STATE_RE.match(line)
        if m and cur is not None:
            cur["states"].append((m.group(1), m.group(2), m.group(3)))
            continue
        m = RUN_RE.match(line)
        if m:
            exe, kind = m.group(1), m.group(2)
            rm = re.search(r"--rows (\d+)", m.group(3))
            rows = int(rm.group(1)) if rm else 17408
            if exe not in exes:
                exes.append(exe)
            continue
        if cur is None:
            continue
        if kind == "membw":
            m = MEMBW_RE.match(line)
            if m:
                cur["membw"][int(m.group(1))] = float(m.group(2))
        elif kind == "kernels":
            m = KERNEL_RE.match(line)
            if m:
                cur["kernels"][(exe, rows)].append((m.group(1), m.group(2), int(m.group(4)), float(m.group(6))))

    # per kernel across rounds: {(exe, rows, kernel, act, thr): [(gbs, pct)]}
    across = defaultdict(list)
    for r in rounds:
        ceiling = max(r["membw"].get(t, 0.0) for t in (6, 12)) if r["membw"] else float("nan")
        print("== round %d: membw best %s -> ceiling %.2f GB/s" % (r["n"], " / ".join("%d thr %.2f" % (t, v) for t, v in sorted(r["membw"].items())), ceiling))
        for label, when, text in r["states"]:
            print("   state [%s] %s: %s" % (label, when, text))
        for (e, rows_), lines in r["kernels"].items():
            tag = "exe %d" % (exes.index(e) + 1)
            print("   %s (%s), %d x 5120:" % (tag, e, rows_))
            for kernel, act, thr, gbs in lines:
                if thr not in threads:
                    continue
                pct = 100.0 * gbs / ceiling if ceiling else float("nan")
                across[(exes.index(e) + 1, rows_, kernel, act, thr)].append((gbs, pct))
                print("      %-5s %-5s %2d thr  %6.2f GB/s  %5.1f %%" % (kernel, act, thr, gbs, pct))
    print()
    print("== range over %d rounds (GB/s, %% of that round's membw); exe 1 = %s" % (len(rounds), exes[0] if exes else "?") + ("; exe 2 = %s" % exes[1] if len(exes) > 1 else ""))
    print("%-5s %-6s %-6s %-5s %3s   %-17s %-15s" % ("exe", "rows", "kernel", "act", "thr", "GB/s min..max", "% min..max"))
    for key in sorted(across):
        vals = across[key]
        g = [v[0] for v in vals]
        p = [v[1] for v in vals]
        print("%-5d %-6d %-6s %-5s %3d   %6.2f .. %6.2f   %5.1f .. %5.1f" % (key[0], key[1], key[2], key[3], key[4], min(g), max(g), min(p), max(p)))


if __name__ == "__main__":
    main()
