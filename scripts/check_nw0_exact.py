"""CPU-only check: does boltz_nw0_exact's emulation give the same batch as real workers?
Usage: check_nw0_exact.py <boltz_results_dir>
"""
import sys
from pathlib import Path

import torch
from rdkit import Chem

import boltz_nw0_exact  # noqa: F401  (installs the single-process patch)
from diag_batch import first_batch


def main():
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    res = Path(sys.argv[1])
    torch.manual_seed(0)
    a = first_batch(res, 0)   # patched main-process featurization
    torch.manual_seed(0)
    b = first_batch(res, 2)   # real worker processes
    bad = [k for k in a if torch.is_tensor(a[k]) and not (
        a[k].shape == b[k].shape and torch.equal(a[k], b[k])
        and a[k].dtype == b[k].dtype and a[k].stride() == b[k].stride())]
    n = sum(torch.is_tensor(v) for v in a.values())
    print(f"{n - len(bad)} of {n} tensors identical; differing: {bad}")


if __name__ == "__main__":
    main()
