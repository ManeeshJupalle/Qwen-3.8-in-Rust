#!/usr/bin/env bash
# scripts/ladder.sh -- the Phase 4 ladder on Linux. UNTESTED: written alongside scripts/ladder.ps1 (the one
# that produced docs/data/ladder.txt) and never run; the engine's Linux read path (O_DIRECT, os.rs) is also
# untested. Kept so the mechanism is on record.
#
# The cap is `systemd-run --scope -p MemoryMax=<budget>` around each run (cgroup v2 memory.max: the kernel
# reclaims and then OOM-kills the process past the limit, so a run that finishes stayed under it). The
# engine still sizes its plan with --budget; --job-limit is Windows-only and is not passed here. Peak RSS is
# the engine's own VmHWM (peak_rss in the stats JSON).
#
#   scripts/ladder.sh [exe] [model] [membw GB/s] [diskbw GB/s]
set -euo pipefail
EXE="${1:-target/release/aqueduct}"
MODEL="${2:-$HOME/models/Qwen3.8-27B-Q4_K_M.gguf}"
MEMBW="${3:-0}"
DISKBW="${4:-0}"
DOCTOR="docs/data/doctor_$(hostname | tr 'A-Z' 'a-z').txt"
if [ "$MEMBW" = "0" ] && [ -f "$DOCTOR" ]; then
  MEMBW=$(grep -E '^membw' "$DOCTOR" | sed -E 's/^membw *: *([0-9.]+) GB.*/\1/')
  DISKBW=$(grep -E '^disk qd2' "$DOCTOR" | sed -E 's/^disk qd2 *: *best ([0-9.]+) GB.*/\1/')
fi
[ "$MEMBW" != "0" ] && [ "$DISKBW" != "0" ] || { echo "need membw and diskbw (run: $EXE doctor)"; exit 2; }
OUT=docs/data/ladder.txt
TMP=$(mktemp -d)
python3 - <<'EOF' > "$TMP/prompts.txt"
import json
for p in json.load(open("tests/fixtures/prompts.json"))["prompts"]:
    print(p["name"], ",".join(str(i) for i in p["ids"]))
EOF
echo "# aqueduct ladder (linux, systemd-run MemoryMax): $(hostname), $(date -u +%F' '%R)" > "$OUT"
for B in 6G 8G 12G 16G resident; do
  while read -r NAME IDS; do
    STATS="$TMP/stats_${B}_${NAME}.json"
    ARGS=(run --model "$MODEL" --ids "$IDS" --ids-only --max-tokens 32 --stats "$STATS" --membw "$MEMBW" --diskbw "$DISKBW")
    if [ "$B" != "resident" ]; then
      ARGS+=(--budget "$B")
      systemd-run --scope --user -p "MemoryMax=$B" -- "$EXE" "${ARGS[@]}" > "$TMP/out_${B}_${NAME}.txt"
    else
      "$EXE" "${ARGS[@]}" > "$TMP/out_${B}_${NAME}.txt"
    fi
    python3 - "$B" "$NAME" "$STATS" "$TMP/out_${B}_${NAME}.txt" "$MEMBW" >> "$OUT" <<'EOF'
import json, sys
b, name, stats, out, membw = sys.argv[1:6]
s = json.load(open(stats))
ids = open(out).read().strip().splitlines()[-1]
print(f"{b:<9} {name:<9} pinned {s['pinned']:3} streamed {s['streamed']:3} disk {s['disk_bytes_per_token']/1e9:.3f} GB/tok "
      f"{s['s_per_token']:.3f} s/tok {s['ram_gbps']:.2f} GB/s {100*s['ram_gbps']/float(membw):.0f}% membw "
      f"pred {s['predicted_s_per_token']} peakRSS {s['peak_rss']/1e9:.3f} GB prefetch {s['prefetch_hidden']:.2f} ids {ids}")
EOF
  done < "$TMP/prompts.txt"
done
echo "filed to $OUT (identity: compare the ids columns against tests/fixtures/ladder_expected_ids.json)"
