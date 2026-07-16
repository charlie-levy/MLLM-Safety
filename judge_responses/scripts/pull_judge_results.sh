#!/bin/bash
# pull_judge_results.sh — Pull the LLaMA-Guard + WildGuard judge outputs from
# Newton into THIS folder (judge_responses/), PROVE every model_responses/ record
# was judged by both guards, and rebuild the ASR table locally (LG/WG/Both/Either).
#
# Run on your Mac AFTER the grid_guard jobs finish:
#     squeue -u ch169788 | grep grid_guard      # must be EMPTY first
#     bash pull_judge_results.sh                 # 1 password prompt
#
# Produces:
#   judge_responses/llamaguard/<cell>.json (+ .summary.json)   [12 cells]
#   judge_responses/wildguard/ <cell>.json (+ .summary.json)   [12 cells]
#   judge_responses/asr_summary.json     (12-cell table: LG, WG, Both, Either, kappa)
#   judge_responses/report_table.txt     (the printed table)
set -u

SRC="ch169788@newton.ist.ucf.edu:/home/ch169788/llava_cot_eval"
DEST="$(cd "$(dirname "$0")" && pwd)"          # this folder (judge_responses/)
TMP="$DEST/.grid_guard_pull"
REPORT="$HOME/Desktop/REU/llava_cot_eval/code/report_grid_guards.py"

echo "==> pulling results/grid_guard_eval (1 password prompt) ..."
rm -rf "$TMP"; mkdir -p "$TMP"
scp -r "$SRC/results/grid_guard_eval" "$TMP/" || { echo "FAILED: results/grid_guard_eval not found — has the job finished?"; exit 1; }

EVAL="$TMP/grid_guard_eval"
rm -rf "$DEST/llamaguard" "$DEST/wildguard"    # clear any stale partial
[ -d "$EVAL/llamaguard" ] && mv "$EVAL/llamaguard" "$DEST/"
[ -d "$EVAL/wildguard" ]  && mv "$EVAL/wildguard"  "$DEST/"
rm -rf "$TMP"                                   # drop _preflight slice + wrapper

# --- Coverage proof: every model_responses/ record judged by BOTH guards ---
echo ""
python3 - "$DEST" <<'PY'
import json, glob, os, sys
dest = sys.argv[1]
inputs = sorted(glob.glob(os.path.join(dest, "model_responses", "*.json")))
def n(p):
    try:
        with open(p) as f: return len(json.load(f))
    except Exception: return None
print("==> coverage vs model_responses/  (judged / input)")
print("    %-34s %7s %12s %12s" % ("cell", "input", "llamaguard", "wildguard"))
ok = True; ti = tl = tw = 0
for ip in inputs:
    cell = os.path.basename(ip)[:-5]; want = n(ip); ti += want or 0
    row = []
    for j in ("llamaguard", "wildguard"):
        got = n(os.path.join(dest, j, cell + ".json"))
        tl += got or 0 if j == "llamaguard" else 0
        tw += got or 0 if j == "wildguard" else 0
        row.append("%d ok" % got if got == want else ("MISSING" if got is None else "%d/%d GAP" % (got, want)))
        if got != want: ok = False
    print("    %-34s %7d %12s %12s" % (cell, want, row[0], row[1]))
print("    " + "-"*66)
print("    %-34s %7d %12d %12d" % ("TOTAL", ti, tl, tw))
done = ok and len(inputs) == 12 and tl == ti and tw == ti
print("\n" + ("COMPLETE — every response in model_responses/ judged by BOTH guards."
             if done else
             "INCOMPLETE — gaps above. Wait until squeue is empty, then re-run this script."))
sys.exit(0 if done else 1)
PY
rc=$?

# --- Rebuild the ASR table locally from the pulled per-item flags (LG/WG/Both/Either) ---
if [ -f "$REPORT" ]; then
  echo ""
  echo "==> ASR table (built locally from pulled files):"
  python3 "$REPORT" "$DEST" | tee "$DEST/report_table.txt"
else
  echo "  (report script not found at $REPORT — skipping local table build)"
fi
exit $rc
