"""Phase 5.4: the speculative round cost model against the ladder's measurements.

Reads the stats files of the last `scripts/ladder.ps1 -Spec K` run (%TEMP%\\aqueduct-ladder: stats_<rung>_<prompt>.json
for the plain runs, stats_spec_<rung>_<prompt>.json for the --spec runs) and, with membw / diskbw from the doctor
file, prints per rung:

  t_plain      the measured plain s/token (mean over prompts)
  t_round      the measured seconds per speculative round (decode seconds / rounds)
  parts        verify (the k+1-row batch through every layer), snapshot, replay, MTP re-feed, chained drafts
  tokens/round the emitted tokens per round (1 + accepted)

and fits the two constants of the brief's model  t_round ~ bytes_ram / membw_eff + bytes_disk / diskbw + c_mtp x k
+ c_verify x (k + 1):  c_mtp = the mean chained-draft step over all rungs (the MTP block and the head are
resident everywhere), c_verify = (verify per round at the resident rung - the resident plain token) / k, the
marginal compute of one extra batch row where nothing hides it. membw_eff is what the plain token measured
(bytes_ram / (t_plain - bytes_disk / diskbw)), so the RAM and disk terms reproduce t_plain by construction and the
model's whole content is the two constants. The ratio measured / predicted per rung is filed; on the streamed rungs
the extra rows' compute hides under the disk reads, which the sum cannot express, and the ratio says by how much.

Usage: python tools/spec_cost_model.py [--doctor docs/data/doctor_<machine>.txt] [--out docs/data/spec_cost_model.txt] [--rungs 5G,11G,16G,resident]
"""
import argparse
import glob
import json
import os
import platform
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["5G", "6G", "8G", "11G", "12G", "16G", "32G", "resident"]


def main():
    ap = argparse.ArgumentParser()
    machine = platform.node().lower()
    ap.add_argument("--doctor", default=os.path.join(ROOT, "docs", "data", "doctor_%s.txt" % machine))
    ap.add_argument("--dir", default=os.path.join(os.environ.get("TEMP", "/tmp"), "aqueduct-ladder"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "spec_cost_model.txt"))
    ap.add_argument("--membw", type=float, default=0.0)
    ap.add_argument("--diskbw", type=float, default=0.0)
    ap.add_argument("--rungs", default="", help="comma-separated rungs to fit (default: every rung with stats files; the directory keeps older runs' files)")
    a = ap.parse_args()
    only = [r for r in a.rungs.split(",") if r]
    membw, diskbw = a.membw, a.diskbw
    if (membw <= 0 or diskbw <= 0) and os.path.exists(a.doctor):
        for line in open(a.doctor, encoding="utf-8"):
            m = re.match(r"^membw\s*:\s*([0-9.]+) GB/s", line)
            if m and membw <= 0:
                membw = float(m.group(1))
            m = re.match(r"^disk qd2\s*:\s*best ([0-9.]+) GB/s", line)
            if m and diskbw <= 0:
                diskbw = float(m.group(1))
    assert membw > 0 and diskbw > 0, "need --membw and --diskbw or a doctor file"

    rungs = {}
    for f in glob.glob(os.path.join(a.dir, "stats_spec_*.json")):
        name = os.path.basename(f)[len("stats_spec_"):-len(".json")]
        rung, prompt = name.split("_", 1)
        if only and rung not in only:
            continue
        plain = os.path.join(a.dir, "stats_%s_%s.json" % (rung, prompt))
        if not os.path.exists(plain):
            continue
        rungs.setdefault(rung, []).append((json.load(open(plain)), json.load(open(f))))
    if not rungs:
        raise SystemExit("no stats_spec_*.json in %s (run scripts/ladder.ps1 -Spec K first)" % a.dir)
    rows = []
    for rung in sorted(rungs, key=lambda r: ORDER.index(r) if r in ORDER else 99):
        pairs = rungs[rung]
        k = pairs[0][1]["spec_k"]
        t_plain = statistics.mean(p["s_per_token"] for p, _ in pairs)
        bytes_disk = pairs[0][0]["streamed_bytes_per_pass"]
        bytes_ram = pairs[0][0]["decode_bytes_per_token"] - bytes_disk
        rounds = sum(s["spec_rounds"] for _, s in pairs)
        t_round = sum(s["decode_s"] for _, s in pairs) / rounds
        verify = sum(s["spec_verify_s"] for _, s in pairs) / rounds
        snap = sum(s["spec_snapshot_s"] for _, s in pairs) / rounds
        replay = sum(s["spec_replay_s"] for _, s in pairs) / rounds
        refeed = sum(s["spec_refeed_s"] for _, s in pairs) / max(1, sum(s["spec_refeeds"] for _, s in pairs))
        chain = sum(s["spec_chain_s"] for _, s in pairs) / max(1, sum(s["spec_chain_steps"] for _, s in pairs))
        tok_round = sum(s["n_generated"] for _, s in pairs) / rounds
        acc = sum(s["spec_accepted"] for _, s in pairs) / rounds
        t_spec = statistics.mean(s["s_per_token"] for _, s in pairs)
        disk_term = bytes_disk / 1e9 / diskbw
        membw_eff = bytes_ram / 1e9 / max(1e-9, t_plain - disk_term)
        rows.append(dict(rung=rung, k=k, t_plain=t_plain, t_spec=t_spec, bytes_ram=bytes_ram, bytes_disk=bytes_disk, rounds=rounds, t_round=t_round, verify=verify, snap=snap, replay=replay,
                         refeed=refeed, chain=chain, tok_round=tok_round, acc=acc, pinned=pairs[0][1]["pinned"], pinned_plain=pairs[0][0]["pinned"], disk_term=disk_term, membw_eff=membw_eff))
    res = next((r for r in rows if r["rung"] in ("resident", "32G")), rows[-1])
    k = res["k"]
    c_mtp = statistics.mean(r["chain"] for r in rows)
    c_verify = (res["verify"] - res["t_plain"]) / k
    L = []
    P = L.append
    P("# spec cost model (tools/spec_cost_model.py) from the --spec %d ladder stats in %s; membw %.2f GB/s, disk qd2 %.2f GB/s" % (k, a.dir, membw, diskbw))
    P("# model: t_round = bytes_ram / membw_eff + bytes_disk / diskbw + c_mtp x k + c_verify x (k + 1)")
    P("#   c_mtp    = %.4f s  (mean chained-draft step over the rungs: MTP block + shared head, both resident)" % c_mtp)
    P("#   c_verify = %.4f s  (fitted at %s: (verify per round %.3f - plain token %.3f) / k = the marginal compute of one extra batch row)" % (c_verify, res["rung"], res["verify"], res["t_plain"]))
    P("#   membw_eff per rung = bytes_ram / (t_plain - bytes_disk / diskbw): the plain token's own RAM rate, so the first two terms reproduce t_plain")
    P("# note: refeed (the MTP over the m+1 accepted rows, head once) and replay/snapshot are measured but not in the model; 'pred+' adds them")
    P("%-9s %5s %9s %9s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %9s %7s" % ("rung", "pin", "t_plain", "t_spec", "t_round", "verify", "snap", "replay", "refeed", "chain", "tok/rnd", "pred", "ratio", "pred+", "ratio+", "membw_eff", "speedup"))
    md = ["| rung | pinned (spec) | plain s/token | spec s/token | speedup | s/round measured | verify | MTP re-feed | chain/step | replay | tokens/round | predicted s/round | measured / predicted |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pred = r["bytes_ram"] / 1e9 / r["membw_eff"] + r["disk_term"] + c_mtp * r["k"] + c_verify * (r["k"] + 1)
        pred_plus = pred + r["refeed"] + r["replay"] + r["snap"]
        ratio = r["t_round"] / pred
        ratio_plus = r["t_round"] / pred_plus
        P("%-9s %5s %9.3f %9.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f %8.3f %8.2f %8.3f %8.2f %9.2f %7.2f" % (r["rung"], "%d/%d" % (r["pinned_plain"], r["pinned"]), r["t_plain"], r["t_spec"], r["t_round"], r["verify"], r["snap"], r["replay"], r["refeed"], r["chain"], r["tok_round"], pred, ratio, pred_plus, ratio_plus, r["membw_eff"], r["t_plain"] / r["t_spec"]))
        md.append("| %s | %d (%d) | %.3f | %.3f | %.2f x | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %.2f |" % (r["rung"], r["pinned_plain"], r["pinned"], r["t_plain"], r["t_spec"], r["t_plain"] / r["t_spec"], r["t_round"], r["verify"], r["refeed"], r["chain"], r["replay"], r["tok_round"], pred, ratio))
    P("")
    P("# markdown for docs/ladder.md")
    L.extend(md)
    text = "\n".join(L) + "\n"
    open(a.out, "w", encoding="utf-8", newline="\n").write(text)
    print(text)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
