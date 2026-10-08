"""Diagnose why the 2nd+ job in a persistent process is not bit-identical to a fresh process.

Runs one input twice in one process (cached model) and fingerprints every parameter and buffer
of the model right after load, after the first predict, and after the second.
Usage: diag_persistent.py <yaml> <out_root> -- <common boltz predict args>
"""
import hashlib
import sys

import boltz.main as bm
import torch
from boltz.model.models.boltz2 import Boltz2

_orig_load = Boltz2.load_from_checkpoint
STATE = {}


def fingerprint(m):
    out = {}
    for name, t in list(m.named_parameters()) + list(m.named_buffers()):
        a = t.detach().to("cpu").contiguous().reshape(-1)
        raw = a.numpy().tobytes() if a.dtype == torch.bool else a.view(torch.uint8).numpy().tobytes()
        out[name] = (str(a.dtype), str(t.device), hashlib.sha256(raw).hexdigest()[:16])
    return out


def cached_load(*a, **k):
    if "m" not in STATE:
        STATE["m"] = _orig_load(*a, **k)
        STATE["fp_load"] = fingerprint(STATE["m"])
        STATE["training"] = STATE["m"].training
    return STATE["m"]


Boltz2.load_from_checkpoint = staticmethod(cached_load)


def diff(a, b, label):
    changed = [k for k in a if a[k] != b.get(k)]
    print(f"{label}: {len(changed)} of {len(a)} tensors changed")
    for k in changed[:15]:
        print(f"   {k}: {a[k]} -> {b[k]}")


def main():
    yaml, out_root = sys.argv[1], sys.argv[2]
    common = sys.argv[sys.argv.index("--") + 1:]
    fps = []
    for i in range(2):
        bm.predict.main(args=[yaml, "--out_dir", f"{out_root}/run{i}", *common],
                        standalone_mode=False)
        fps.append(fingerprint(STATE["m"]))
        print(f"after run{i}: training={STATE['m'].training} (at load: {STATE['training']})")
    diff(STATE["fp_load"], fps[0], "load -> after run0")
    diff(fps[0], fps[1], "after run0 -> after run1")


# Guard required: macOS dataloader workers use spawn and re-import this file.
if __name__ == "__main__":
    main()
