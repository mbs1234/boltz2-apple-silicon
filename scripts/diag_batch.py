"""Does the dataloader worker count change the batch the model sees?

Builds the first predict batch for one processed input with num_workers=0 and num_workers=2
and compares every tensor: values, dtype, shape, strides, contiguity.
Usage: diag_batch.py <boltz_results_dir>
"""
import os
import sys
from pathlib import Path

import torch
from rdkit import Chem
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest


def first_batch(res, nw):
    p = res / "processed"
    dm = Boltz2InferenceDataModule(
        manifest=Manifest.load(p / "manifest.json"), target_dir=p / "structures",
        msa_dir=p / "msa", mol_dir=Path(os.environ.get("BOLTZ2_HOME", Path.home() / "boltz2-dev")) / "cache" / "mols",
        num_workers=nw, constraints_dir=p / "constraints", template_dir=p / "templates",
        extra_mols_dir=p / "mols", override_method=None)
    return next(iter(dm.predict_dataloader()))


def main():
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)  # as boltz predict does
    res = Path(sys.argv[1])
    a, b = first_batch(res, 0), first_batch(res, 2)
    diffs = 0
    for k in sorted(a):
        x, y = a[k], b[k]
        if not torch.is_tensor(x):
            continue
        same_val = x.shape == y.shape and torch.equal(x, y)
        same_layout = (x.dtype == y.dtype and x.stride() == y.stride()
                       and x.is_contiguous() == y.is_contiguous())
        if not (same_val and same_layout):
            diffs += 1
            print(f"{k}: values_equal={same_val} dtype {x.dtype}/{y.dtype} "
                  f"stride {x.stride()}/{y.stride()} contig {x.is_contiguous()}/{y.is_contiguous()}")
    n = sum(torch.is_tensor(v) for v in a.values())
    print(f"{diffs} of {n} tensors differ between num_workers=0 and num_workers=2")


if __name__ == "__main__":
    main()


def detail(res):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    a, b = first_batch(res, 0), first_batch(res, 2)
    d = (a["ref_pos"] - b["ref_pos"]).norm(dim=-1)[0]
    mask = a["atom_pad_mask"][0].bool()
    # map atoms -> token -> mol_type (0 protein, 3 nonpolymer)
    tok = a["atom_to_token"][0].float().argmax(-1)
    mt = a["mol_type"][0][tok]
    for t, label in ((0, "protein"), (3, "ligand")):
        sel = mask & (mt == t)
        dd = d[sel]
        print(f"{label}: atoms {int(sel.sum())}, differing {int((dd > 0).sum())}, "
              f"max |d| {float(dd.max()):.4g} A, mean |d| {float(dd.mean()):.4g} A")
