"""Export one baseline pass to a flat per-sample CSV for pickup by the lab.

Usage: export_samples.py <pass_dir>   -> writes <pass_dir>/samples.csv

One row per (input, model rank). Confidence fields are boltz's own names, unrenamed;
wall-clock and memory come from the pass's times.tsv (per job, repeated on each sample row).
`model` is boltz's rank (model_0 = highest confidence_score), not the diffusion draw order.
"""
import csv
import json
import sys
from pathlib import Path

FIELDS = ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_plddt",
          "complex_iplddt", "complex_pde", "complex_ipde"]

pass_dir = Path(sys.argv[1])
times = {r["input"]: r for r in csv.DictReader(open(pass_dir / "times.tsv"), delimiter="\t")}
rows = []
for pred in sorted(pass_dir.glob("boltz_results_*/predictions/*")):
    name = pred.name
    uuid8, construct, ligand = name.split("_")
    t = times.get(name, {})
    for conf_path in sorted(pred.glob(f"confidence_{name}_model_*.json")):
        k = int(conf_path.stem.rsplit("_", 1)[1])
        conf = json.loads(conf_path.read_text())
        rows.append({"input": name, "ref_uuid8": uuid8, "construct": construct, "ligand": ligand,
                     "seed": t.get("seed"), "model": k,
                     **{f: conf.get(f) for f in FIELDS},
                     "job_wall_s": t.get("wall_s"), "job_peak_footprint_bytes":
                     t.get("peak_footprint_bytes"),
                     "cif": str((pred / f"{name}_model_{k}.cif").relative_to(pass_dir))})
with open(pass_dir / "samples.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {pass_dir / 'samples.csv'}")
