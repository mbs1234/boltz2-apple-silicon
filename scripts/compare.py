"""Compare Boltz-2 prediction sets: same-seed repeatability, between-seed noise floor, and
candidate-vs-baseline deviation.

Usage:
  compare.py noise  <run_dir> [<run_dir> ...]        # within-job spread pooled over seeds
  compare.py paired <baseline_dir> <candidate_dir>   # same seed, sample-by-sample deviation

A run_dir is runs/<...>/<pass>, holding boltz_results_<input>/predictions/<input>/.
Structures are compared in the protein frame: superpose on protein CA, then report protein CA
RMSD and ligand heavy-atom RMSD *without* re-fitting on the ligand (pose displacement in the
pocket, which is what a triage decision reads). `lig_rmsd_pocket` is name-matched (an upper
bound for symmetric ligands); `lig_rmsd_pocket_sym` reassigns same-element atoms optimally
(Hungarian) first, so symmetric groups such as CF3 or a flipped ring do not count as error.
Element-only matching ignores bonding, so it can also pair chemically distinct atoms: treat the
two as a bracket, name-matched = upper bound, sym = lower bound.
"""
import itertools
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment

CONF_KEYS = ["confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt",
             "complex_iplddt", "complex_pde", "complex_ipde"]


def load_structure(cif, with_elements=False):
    st = gemmi.read_structure(str(cif))
    ca, lig, el = [], [], []
    for ch in st[0]:
        for res in ch:
            for at in res:
                if ch.name == "A" and at.name == "CA":
                    ca.append(at.pos.tolist())
                elif ch.name != "A" and at.element.name != "H":
                    lig.append(at.pos.tolist())
                    el.append(at.element.name)
    if with_elements:
        return np.array(ca), np.array(lig), np.array(el)
    return np.array(ca), np.array(lig)


def sym_rmsd(a, b, elements):
    """Ligand RMSD after optimal same-element reassignment (Hungarian, per element).

    Removes the inflation from symmetric groups (CF3, flipped rings) that a name-matched RMSD
    counts as error. It can only be <= the name-matched RMSD.
    """
    sq = 0.0
    for e in np.unique(elements):
        idx = np.where(elements == e)[0]
        cost = ((a[idx][:, None, :] - b[idx][None, :, :]) ** 2).sum(2)
        r, c = linear_sum_assignment(cost)
        sq += cost[r, c].sum()
    return float(np.sqrt(sq / len(a)))


def kabsch(P, Q):
    """Rotation R and translation t minimising |(P @ R.T + t) - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


POCKET_A = 10.0  # CA within this distance of any ligand heavy atom (in structure a) = pocket


def struct_dev(cif_a, cif_b):
    ca_a, lig_a, el = load_structure(cif_a, with_elements=True)
    ca_b, lig_b = load_structure(cif_b)
    out = {"identical": bool(np.array_equal(ca_a, ca_b) and np.array_equal(lig_a, lig_b))}
    R, t = kabsch(ca_b, ca_a)
    out["ca_rmsd"] = 0.0 if out["identical"] else rmsd(ca_b @ R.T + t, ca_a)
    if len(lig_a):
        out["lig_rmsd"] = 0.0 if out["identical"] else rmsd(lig_b @ R.T + t, lig_a)
        d = np.linalg.norm(ca_a[:, None, :] - lig_a[None, :, :], axis=2).min(1)
        pk = d < POCKET_A
        Rp, tp = kabsch(ca_b[pk], ca_a[pk])
        out["pocket_ca_rmsd"] = 0.0 if out["identical"] else rmsd(ca_b[pk] @ Rp.T + tp, ca_a[pk])
        out["lig_rmsd_pocket"] = 0.0 if out["identical"] else rmsd(lig_b @ Rp.T + tp, lig_a)
        out["lig_rmsd_pocket_sym"] = (0.0 if out["identical"]
                                      else sym_rmsd(lig_a, lig_b @ Rp.T + tp, el))
    return out


def samples(run_dir):
    """{input: [(cif, conf_dict), ...]} ordered by boltz's model rank."""
    res = {}
    for pred in sorted(Path(run_dir).glob("boltz_results_*/predictions/*")):
        name = pred.name
        items = []
        for k in range(100):
            cif = pred / f"{name}_model_{k}.cif"
            if not cif.exists():
                break
            conf = json.loads((pred / f"confidence_{name}_model_{k}.json").read_text())
            items.append((cif, conf))
        res[name] = items
    return res


def paired(base_dir, cand_dir):
    A, B = samples(base_dir), samples(cand_dir)
    rows = []
    for name in sorted(A):
        if name not in B:
            print(f"MISSING in candidate: {name}", file=sys.stderr)
            continue
        for k, ((ca, fa), (cb, fb)) in enumerate(zip(A[name], B[name])):
            r = {"input": name, "model": k, **struct_dev(ca, cb)}
            for key in CONF_KEYS:
                if key in fa and key in fb:
                    r["d_" + key] = fb[key] - fa[key]
            rows.append(r)
    return rows


def noise(run_dirs):
    per = {}
    for d in run_dirs:
        for name, items in samples(d).items():
            per.setdefault(name, []).extend(items)
    rows = []
    for name, items in sorted(per.items()):
        conf = {k: np.array([c[k] for _, c in items if k in c]) for k in CONF_KEYS}
        pair = [struct_dev(a[0], b[0]) for a, b in itertools.combinations(items, 2)]
        r = {"input": name, "n_samples": len(items)}
        for k, v in conf.items():
            if len(v):
                r[k + "_sd"] = float(v.std(ddof=1))
                r[k + "_range"] = float(v.max() - v.min())
        r["pair_ca_rmsd_median"] = float(np.median([p["ca_rmsd"] for p in pair]))
        r["pair_ca_rmsd_max"] = float(max(p["ca_rmsd"] for p in pair))
        for key in ("lig_rmsd", "lig_rmsd_pocket", "lig_rmsd_pocket_sym", "pocket_ca_rmsd"):
            if key in pair[0]:
                v = [p[key] for p in pair]
                r[f"pair_{key}_median"] = float(np.median(v))
                r[f"pair_{key}_max"] = float(max(v))
        rows.append(r)
    return rows


def summarize_paired(rows):
    def mx(key):
        v = [abs(r[key]) for r in rows if key in r]
        return max(v) if v else None
    return {"n_pairs": len(rows), "n_identical": sum(r["identical"] for r in rows),
            "max_ca_rmsd": mx("ca_rmsd"), "max_lig_rmsd": mx("lig_rmsd"),
            "max_pocket_ca_rmsd": mx("pocket_ca_rmsd"), "max_lig_rmsd_pocket": mx("lig_rmsd_pocket"),
            "max_lig_rmsd_pocket_sym": mx("lig_rmsd_pocket_sym"),
            **{"max_abs_d_" + k: mx("d_" + k) for k in CONF_KEYS}}


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "paired":
        rows = paired(sys.argv[2], sys.argv[3])
        print(json.dumps({"summary": summarize_paired(rows), "rows": rows}, indent=1))
    elif mode == "noise":
        rows = noise(sys.argv[2:])
        print(json.dumps(rows, indent=1))
    else:
        sys.exit(__doc__)
