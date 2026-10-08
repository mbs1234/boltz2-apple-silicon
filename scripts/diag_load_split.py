"""Split Boltz2.load_from_checkpoint wall time into: torch.load of the file, model construction
(__init__, incl. random weight init the checkpoint then overwrites), and load_state_dict.
Also reports how much CPU RNG the construction consumes (whether a skip-init must restore it).
Runs one real `boltz predict` and stops at the first predict_step.
Usage: diag_load_split.py -- <boltz predict args...>
"""
import sys
import time

import torch

import boltz_nw0_exact  # noqa: F401

T = {}


class _Stop(Exception):
    pass


def timed(label, fn):
    def inner(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[label] = T.get(label, 0.0) + time.perf_counter() - t
    return inner


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    import boltz.main as bm
    import pytorch_lightning.core.saving as saving
    from boltz.model.models.boltz2 import Boltz2

    saving.pl_load = timed("torch.load (read checkpoint)", saving.pl_load)
    orig_lsd = torch.nn.Module.load_state_dict
    torch.nn.Module.load_state_dict = timed("load_state_dict", orig_lsd)
    orig_lfc = Boltz2.load_from_checkpoint

    def lfc(*a, **k):
        before = torch.get_rng_state().clone()
        t = time.perf_counter()
        out = orig_lfc(*a, **k)
        T["load_from_checkpoint (total)"] = time.perf_counter() - t
        # file read and load_state_dict draw no random numbers, so any change is construction
        T["cpu RNG changed during load"] = not torch.equal(before, torch.get_rng_state())
        return out

    Boltz2.load_from_checkpoint = staticmethod(lfc)

    def stop(self, *a, **k):
        raise _Stop

    Boltz2.predict_step = stop
    try:
        bm.predict.main(args=[*argv, "--num_workers", "0"], standalone_mode=False)
    except _Stop:
        pass
    T["construction (= total - read - load_state_dict)"] = (
        T["load_from_checkpoint (total)"] - T["torch.load (read checkpoint)"] - T["load_state_dict"])
    for k, v in T.items():
        print(f"{v:8.2f} s  {k}" if isinstance(v, float) else f"{v!s:>8}    {k}")


if __name__ == "__main__":
    main()
