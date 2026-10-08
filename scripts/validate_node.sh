#!/bin/bash
# Acceptance test for a newly deployed node: does it reproduce the reference archive?
#
# Run on the new machine after following DEPLOY.md, from a logged-in session (not over SSH:
# MPS and /Volumes access differ there). Takes ~25 min of GPU time. Nothing else may use the GPU.
#
# Gates, in order (stops at the first failure):
#   G0  `cofold` fingerprint unchanged (stop-work condition if not)
#   G1  boltz is the fixes commit, from a clean git checkout (fast_load and provenance need it)
#   G2  import-time versions equal provenance/boltzfix-import-versions.txt (except the macos line)
#   G3  MPS parity probe passes (scripts/mps_probe.py)
#   G4  weights sha256 equal provenance/weights.json
#   G5  fixed set, seed 0, unmodified CLI (one process per job)  vs archive of record
#   G6  fixed set, seed 0, optimized single-job mode              vs archive of record
#   G7  fixed set, seed 0, optimized batch mode                   vs archive of record
# G5-G7 PASS = 45/45 structures and 135/135 pLDDT/PAE/PDE arrays bit-identical.
# If G5 is not bit-identical but G0-G4 pass, the node differs numerically from the reference node (e.g. a
# different macOS build or GPU core count). Do NOT use it; the report includes the distributional
# comparison to take to the lab.
#
# Usage: scripts/validate_node.sh [<label>]      (env: BOLTZ2_HOME, CONDA_ENVS, BOLTZ_ENV)
set -uo pipefail
D="${BOLTZ2_HOME:-$HOME/boltz2-dev}"
ENVS="${CONDA_ENVS:-$HOME/miniforge3/envs}"
BENV="${BOLTZ_ENV:-boltzfix}"
PY="$ENVS/$BENV/bin/python"
BOLTZ="$ENVS/$BENV/bin/boltz"
CONDA="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
REF="$D/runs/fixes/fix3_9632631_seed0"
EXPECTED_COMMIT=9632631bcc327cee8dfbd330e625bcdfa1f85015
EXPECTED_COFOLD=ff8d2dda6a69a48197eb8cb1b1d0f34a1913577d10d70b1d033e66d0e9780e06
LABEL="${1:-$(hostname -s)-$(date +%Y%m%d-%H%M)}"
OUT="$D/runs/validate/$LABEL"
mkdir -p "$OUT"
REPORT="$OUT/REPORT.txt"
say() { echo "$*" | tee -a "$REPORT"; }
fail() { say "FAIL: $*"; say "RESULT: NODE NOT ACCEPTED"; exit 1; }

say "# validate_node $LABEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)  host $(hostname)  macOS $(sw_vers -productVersion) $(sw_vers -buildVersion)"

# G0
h=$("$CONDA" env export --no-builds -n cofold 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
[ "$h" = "$EXPECTED_COFOLD" ] || fail "G0 cofold fingerprint $h != $EXPECTED_COFOLD (stop-work: report to the lab)"
say "G0 ok  cofold fingerprint unchanged"

# G1
src=$("$PY" -c "import boltz,os;print(os.path.dirname(boltz.__file__))") || fail "G1 cannot import boltz from $BENV"
repo=$(cd "$src/../.." && pwd)
c=$(git -C "$repo" rev-parse HEAD 2>/dev/null) || fail "G1 boltz at $repo is not a git checkout"
[ "$c" = "$EXPECTED_COMMIT" ] || fail "G1 boltz commit $c != $EXPECTED_COMMIT"
[ -z "$(git -C "$repo" status --porcelain)" ] || fail "G1 boltz checkout $repo has local modifications"
say "G1 ok  boltz $c (clean) at $repo"

# G2
"$PY" -c "
import numpy, scipy, torch, pytorch_lightning as pl, rdkit, numba, sklearn, sys, platform
print(f'python {sys.version.split()[0]}\nnumpy {numpy.__version__}\nscipy {scipy.__version__}\ntorch {torch.__version__}\npytorch_lightning {pl.__version__}\nrdkit {rdkit.__version__}\nnumba {numba.__version__}\nscikit-learn {sklearn.__version__}\nmacos {platform.mac_ver()[0]}')" > "$OUT/import-versions.txt"
if ! diff <(grep -v '^macos' "$D/provenance/boltzfix-import-versions.txt") <(grep -v '^macos' "$OUT/import-versions.txt") > "$OUT/import-versions.diff"; then
  fail "G2 import-time versions differ (see $OUT/import-versions.diff)"
fi
say "G2 ok  import-time versions match ($(grep macos "$OUT/import-versions.txt"); reference node was 27.0.1)"

# G3
"$PY" "$D/scripts/mps_probe.py" > "$OUT/mps-probe.json" 2>&1
grep -q '"parity_pass": true' "$OUT/mps-probe.json" || fail "G3 MPS parity probe failed (see $OUT/mps-probe.json)"
say "G3 ok  MPS parity ($(grep mps_gemm_tflops "$OUT/mps-probe.json" | tr -d ' ,'))"

# G4
"$PY" - "$D" > "$OUT/weights-check.txt" 2>&1 <<'PY' || fail "G4 weights check failed (see $OUT/weights-check.txt)"
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); rec = json.loads((d / "provenance/weights.json").read_text())
bad = 0
for f, m in rec["files"].items():
    h = hashlib.sha256()
    with open(d / "cache" / f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    ok = h.hexdigest() == m["sha256"]; bad += not ok
    print(f, "ok" if ok else f"MISMATCH {h.hexdigest()}")
sys.exit(1 if bad else 0)
PY
say "G4 ok  weights sha256 match provenance/weights.json"

check() {  # $1 = run dir, $2 = gate name
  "$PY" "$D/scripts/compare.py" paired "$REF" "$1" > "$1/vs_archive.json" 2>/dev/null
  "$PY" - "$REF" "$1" <<'PY'
import glob, json, sys
import numpy as np
ref, cand = sys.argv[1], sys.argv[2]
s = json.load(open(f"{cand}/vs_archive.json"))["summary"]
n = same = 0
for a in sorted(glob.glob(f"{ref}/boltz_results_*/predictions/*/*.npz")):
    b = a.replace(ref, cand); n += 1
    try:
        xa, xb = np.load(a), np.load(b)
        same += all(np.array_equal(xa[k], xb[k]) for k in xa.files)
    except FileNotFoundError:
        pass
ok = s["n_identical"] == 45 and s["n_pairs"] == 45 and same == n == 135
print(f"{'PASS' if ok else 'FAIL'} structures {s['n_identical']}/{s['n_pairs']} npz {same}/{n} "
      f"max_lig_rmsd_pocket {s['max_lig_rmsd_pocket']:.3f} max_dconf {s['max_abs_d_confidence_score']:.4f}")
sys.exit(0 if ok else 1)
PY
}

COMMON=(--cache "$D/cache" --diffusion_samples 5 --accelerator gpu --no_kernels --seed 0)

# G5 unmodified CLI
mkdir -p "$OUT/g5_stock"
t=$(date +%s)
for Y in "$D"/inputs/yaml/*.yaml; do "$BOLTZ" predict "$Y" --out_dir "$OUT/g5_stock" "${COMMON[@]}" >> "$OUT/g5_stock/log.txt" 2>&1 || fail "G5 boltz predict failed on $Y"; done
r=$(check "$OUT/g5_stock" G5); rc=$?
say "G5 $r  wall $(( $(date +%s) - t )) s (reference: 925.7-934.1 s)"
if [ $rc -ne 0 ]; then
  "$PY" "$D/scripts/compare_sets.py" "$OUT/g5_vs_archive_sets.json" --ref "$D"/runs/fixes/fix3_9632631_seed{0,1,2} --test "$OUT/g5_stock" >> "$REPORT" 2>/dev/null
  fail "G5 node does not reproduce the archive bit-for-bit; distributional comparison appended above"
fi

# G6 optimized single-job mode
mkdir -p "$OUT/g6_single"
t=$(date +%s)
for Y in "$D"/inputs/yaml/*.yaml; do
  BOLTZ2_HOME="$D" BOLTZ_EXACT_PATCHES=1 BOLTZ_FAST_LOAD=1 "$PY" "$D/scripts/boltz_nw0_exact.py" "$Y" --out_dir "$OUT/g6_single" "${COMMON[@]}" >> "$OUT/g6_single/log.txt" 2>&1 || fail "G6 failed on $Y"
done
r=$(check "$OUT/g6_single" G6) || fail "G6 $r"
say "G6 $r  wall $(( $(date +%s) - t )) s (reference: 687.6 s)"

# G7 optimized batch mode
mkdir -p "$OUT/g7_batch"
t=$(date +%s)
(cd "$D/scripts" && BOLTZ2_HOME="$D" BOLTZ_EXACT_PATCHES=1 BOLTZ_OVERLAP=1 BOLTZ_FAST_LOAD=1 "$PY" serve_batch.py "$OUT/g7_batch" "$OUT/g7_batch/times.tsv" "$D"/inputs/yaml/*.yaml -- "${COMMON[@]}" > "$OUT/g7_batch/log.txt" 2>&1) || fail "G7 serve_batch failed (see $OUT/g7_batch/log.txt)"
r=$(check "$OUT/g7_batch" G7) || fail "G7 $r"
say "G7 $r  wall $(( $(date +%s) - t )) s (reference: 585.9 s)"

say "RESULT: NODE ACCEPTED (reproduces the reference archive bit-for-bit in all three modes)"
