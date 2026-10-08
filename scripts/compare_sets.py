"""Distributional comparison of two prediction sets that cannot be paired sample-by-sample
(e.g. different codebases: the same seed is a different random draw in each).

Usage: compare_sets.py <out.json> --ref <run_dir> [<run_dir> ...] --test <run_dir> [...]

Per input:
  * confidence_score / iptm: test mean - ref mean, in units of the ref's own sample SD, and
    whether the test mean lies inside the ref's observed [min, max];
  * pose (pocket frame, ligand heavy atoms): median ligand RMSD over cross-set pairs vs the median
    over within-ref pairs, name-matched and symmetry-tolerant. Equivalent sets give cross ~ within.
"""
import itertools
import json
import sys

import numpy as np

from compare import CONF_KEYS, samples, struct_dev  # noqa: F401


def collect(dirs):
    out = {}
    for d in dirs:
        for name, items in samples(d).items():
            out.setdefault(name, []).extend(items)
    return out


def main():
    out_path = sys.argv[1]
    args = sys.argv[2:]
    ref_dirs = args[args.index("--ref") + 1: args.index("--test")]
    test_dirs = args[args.index("--test") + 1:]
    R, T = collect(ref_dirs), collect(test_dirs)
    rows = []
    for name in sorted(R):
        if name not in T:
            print(f"MISSING in test: {name}", file=sys.stderr)
            continue
        r, t = R[name], T[name]
        row = {"input": name, "n_ref": len(r), "n_test": len(t)}
        for key in ("confidence_score", "iptm"):
            rv = np.array([c[key] for _, c in r])
            tv = np.array([c[key] for _, c in t])
            sd = rv.std(ddof=1)
            row[f"{key}_ref_mean"] = float(rv.mean())
            row[f"{key}_test_mean"] = float(tv.mean())
            row[f"{key}_diff"] = float(tv.mean() - rv.mean())
            row[f"{key}_diff_in_ref_sd"] = float((tv.mean() - rv.mean()) / sd) if sd > 0 else 0.0
            row[f"{key}_test_mean_in_ref_range"] = bool(rv.min() <= tv.mean() <= rv.max())
        within = [struct_dev(a[0], b[0]) for a, b in itertools.combinations(r, 2)]
        cross = [struct_dev(a[0], b[0]) for a in r for b in t]
        for key in ("lig_rmsd_pocket", "lig_rmsd_pocket_sym", "ca_rmsd"):
            if key in within[0]:
                row[f"{key}_within_ref_median"] = float(np.median([w[key] for w in within]))
                row[f"{key}_cross_median"] = float(np.median([c[key] for c in cross]))
        rows.append(row)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    hdr = (f"{'input':28s} {'dConf':>7s} {'dConf/sd':>8s} {'inRng':>5s} {'dIptm':>7s} "
           f"{'dIptm/sd':>8s} | {'ligSym within':>13s} {'cross':>6s} | {'CA within':>9s} {'cross':>6s}")
    print(hdr)
    for w in rows:
        lw = w.get("lig_rmsd_pocket_sym_within_ref_median")
        lc = w.get("lig_rmsd_pocket_sym_cross_median")
        lig = f"{lw:13.2f} {lc:6.2f}" if lw is not None else f"{'-':>13s} {'-':>6s}"
        print(f"{w['input']:28s} {w['confidence_score_diff']:+7.4f} "
              f"{w['confidence_score_diff_in_ref_sd']:+8.2f} "
              f"{str(w['confidence_score_test_mean_in_ref_range']):>5s} {w['iptm_diff']:+7.4f} "
              f"{w['iptm_diff_in_ref_sd']:+8.2f} | {lig} | "
              f"{w['ca_rmsd_within_ref_median']:9.2f} {w['ca_rmsd_cross_median']:6.2f}")


if __name__ == "__main__":
    main()
