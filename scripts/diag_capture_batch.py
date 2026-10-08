"""Capture the first batch that reaches Boltz2.predict_step in a real `boltz predict` run, then
stop before inference. Also records the dataloader base_seed and torch CPU RNG state at
iterator creation.

Usage: diag_capture_batch.py <out.pt> <exact|plain> <boltz predict args...>
  exact -> through boltz_nw0_exact's patch (forces --num_workers 0)
  plain -> unmodified CLI (pass --num_workers yourself if wanted)
"""
import sys

import torch
from torch.utils.data import dataloader as tdl

OUT, MODE = sys.argv[1], sys.argv[2]
if MODE == "exact":
    import boltz_nw0_exact  # noqa: F401

INFO = {"base_seeds": [], "iter_classes": []}
_orig_base_init = tdl._BaseDataLoaderIter.__init__


def _rec_init(self, loader):
    INFO["rng_before_iter"] = torch.get_rng_state().clone()
    _orig_base_init(self, loader)
    INFO["base_seeds"].append(self._base_seed)
    INFO["iter_classes"].append(type(self).__name__)


tdl._BaseDataLoaderIter.__init__ = _rec_init


class _Stop(Exception):
    pass


def main():
    import boltz.main as bm
    from boltz.model.models.boltz2 import Boltz2

    def capture(self, batch, *a, **k):
        cpu = {k2: (v.detach().cpu() if torch.is_tensor(v) else v) for k2, v in batch.items()}
        torch.save({"batch": cpu, **INFO}, OUT)
        print(f"captured -> {OUT}; base_seeds={INFO['base_seeds']} iters={INFO['iter_classes']}")
        raise _Stop

    Boltz2.predict_step = capture
    argv = sys.argv[3:]
    if MODE == "exact":
        argv += ["--num_workers", "0"]
    try:
        bm.predict.main(args=argv, standalone_mode=False)
    except _Stop:
        pass


if __name__ == "__main__":
    main()
