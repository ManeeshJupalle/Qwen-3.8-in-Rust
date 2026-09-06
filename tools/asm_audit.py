"""Audit the x86-64-v3 release build: walk the call graph of the CLI crate's assembly from the C entry `main`
(and from std's `lang_start` closure, which `lang_start_internal` reaches through a vtable rather than a
direct call) to the cpuid gate and list every VEX-encoded (v*) or BMI/LZCNT/MOVBE mnemonic on that path.
Calls into symbols not defined in this crate's assembly are the precompiled standard library (baseline
x86-64). The walk stops at `real_main` (after the gate) and at the first `cpuid` inside `require_avx2`."""
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read().split("\n")

bodies = {}
cur = None
for line in text:
    m = re.match(r"^([A-Za-z_$?@.][\w$?@.]*):", line)
    if m and not line.startswith("."):
        cur = m.group(1)
        bodies.setdefault(cur, [])
        continue
    if cur is not None:
        bodies[cur].append(line)

BMI = {"shlx", "shrx", "sarx", "andn", "bextr", "blsi", "blsmsk", "blsr", "bzhi", "mulx", "pdep", "pext", "rorx", "tzcnt", "lzcnt", "movbe"}
CALL = re.compile(r"^\s*(?:call|callq|jmp|jmpq)\s+([A-Za-z_$?@.][\w$?@.]*)\s*(?:#.*)?$")


def instrs(sym, stop_at_cpuid=False):
    out, calls = [], []
    for l in bodies[sym]:
        s = l.strip()
        if not s or s.startswith(".") or s.startswith("#") or s.startswith(";") or s.endswith(":"):
            continue
        mn = s.split()[0].lower()
        out.append(mn)
        if stop_at_cpuid and mn == "cpuid":
            break
        c = CALL.match(l)
        if c:
            calls.append(c.group(1))
    return out, calls


ok = True
seen = set()
# __rust_begin_short_backtrace calls its argument through a register (`callq *%rcx`); the pointer it is
# handed by the lang_start closure is aqueduct::main, so that is a root too.
roots = ["main"] + [s for s in bodies if "lang_start" in s and "closure" in s] + [s for s in bodies if s.startswith("_ZN8aqueduct4main")]
print("roots:", roots)
todo = list(roots)
reached_gate = False
while todo:
    sym = todo.pop(0)
    if sym in seen:
        continue
    seen.add(sym)
    if sym not in bodies:
        print(f"  {sym}: not in this crate (precompiled std / CRT), not walked")
        continue
    if "real_main" in sym:
        print(f"  {sym}: after the gate, not walked")
        continue
    gate = "require_avx2" in sym
    ins, calls = instrs(sym, stop_at_cpuid=gate)
    bad = sorted({m for m in ins if m.startswith("v") or m in BMI})
    print(f"  {sym}: {len(ins)} instructions{' up to cpuid' if gate else ''}; VEX/BMI: {bad or 'none'}; calls {calls or '-'}")
    if bad:
        ok = False
    if gate:
        reached_gate = "cpuid" in ins
    else:
        todo.extend(c for c in calls if c not in seen)
if not reached_gate:
    ok = False
    print("  the walk did not reach cpuid inside require_avx2")
print("AUDIT", "PASS" if ok else "FAIL", f"({len(seen)} symbols walked)")
