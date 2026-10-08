"""`boltz predict` with featurization in the main process (no dataloader worker processes),
reproducing the num_workers=2 outputs bit-for-bit.

Why this is needed: the featurizer's per-residue reference-conformer roto-translation
(featurizerv2 -> center_random_augmentation) draws from torch's *global* CPU RNG. In a worker
that RNG is seeded by torch's _worker_loop (base_seed + worker_id); in the main process it is
whatever state the main process happens to be in. So plain --num_workers 0 changes inputs.

What this does: before each item is featurized in the main process, swap in the exact RNG
state (torch CPU, python `random`, numpy global) and thread count (1) that worker
`index % EMULATED_WORKERS` would have, continuing that worker's stream across items; then
restore the main process state. Upstream code is not modified; this is a launcher.

Usage: boltz_nw0_exact.py <boltz predict args...>   (do not pass --num_workers; it is forced to 0)
"""
import random
import sys

import numpy as np
import torch
from torch.utils.data import dataloader as tdl
from torch.utils.data._utils.worker import _generate_state

EMULATED_WORKERS = 2  # upstream default --num_workers


class _RNGState:
    def __init__(self):
        self.torch = torch.get_rng_state()
        self.py = random.getstate()
        self.np = np.random.get_state()
        self.threads = torch.get_num_threads()

    def restore(self):
        torch.set_rng_state(self.torch)
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_num_threads(self.threads)


class WorkerEmulatingDataset:
    def __init__(self, dataset, base_seed):
        self.dataset = dataset
        self.base_seed = base_seed
        self.worker_states = {}

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        wid = index % EMULATED_WORKERS
        main_state = _RNGState()
        try:
            if wid in self.worker_states:
                self.worker_states[wid].restore()
            else:  # first item for this worker: seed exactly as torch's _worker_loop does
                seed = self.base_seed + wid
                random.seed(seed)
                # CPU generator only. torch.manual_seed would also reseed the MPS generator,
                # which in a real worker is harmless (separate process) but here would clobber
                # the main process's diffusion-noise stream.
                torch.default_generator.manual_seed(seed)
                np.random.seed(_generate_state(self.base_seed, wid))
            torch.set_num_threads(1)
            item = self.dataset[index]
            self.worker_states[wid] = _RNGState()
        finally:
            main_state.restore()
        return item


_orig_single_init = tdl._SingleProcessDataLoaderIter.__init__


def _patched_single_init(self, loader):
    _orig_single_init(self, loader)
    fetcher = self._dataset_fetcher
    fetcher.dataset = WorkerEmulatingDataset(fetcher.dataset, self._base_seed)


tdl._SingleProcessDataLoaderIter.__init__ = _patched_single_init


def apply_precision_override():
    """BOLTZ_PRECISION=32 replaces upstream's hardcoded bf16-mixed Trainer precision (Phase 2,
    item 3). Unset = upstream behaviour, unchanged."""
    import os

    import boltz.main as bm

    prec = os.environ.get("BOLTZ_PRECISION")
    if not prec:
        return
    base = bm.Trainer

    class Trainer(base):
        def __init__(self, *a, **k):
            k["precision"] = int(prec) if prec.isdigit() else prec
            print(f"[launcher] Trainer precision overridden -> {k['precision']}", flush=True)
            super().__init__(*a, **k)

    bm.Trainer = Trainer


def apply_exact_patches():
    """BOLTZ_EXACT_PATCHES=1 enables scripts/exact_patches.py (Phase 2, item 4)."""
    import os
    if os.environ.get("BOLTZ_EXACT_PATCHES") == "1":
        import exact_patches
        exact_patches.apply()


def apply_fast_load():
    """BOLTZ_FAST_LOAD=1 enables scripts/fast_load.py (skip overwritten init, RNG restored)."""
    import os
    if os.environ.get("BOLTZ_FAST_LOAD") == "1":
        import fast_load
        fast_load.install()


def main():
    import boltz.main as bm
    apply_precision_override()
    apply_exact_patches()
    apply_fast_load()
    argv = [a for a in sys.argv[1:]]
    if "--num_workers" in argv:
        sys.exit("do not pass --num_workers; this launcher forces 0")
    bm.predict.main(args=[*argv, "--num_workers", "0"], standalone_mode=True)


if __name__ == "__main__":
    main()
