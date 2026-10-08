"""Skip the random weight init that the checkpoint overwrites (kit idea `waste_*`), exactly.

Measured: Boltz2.load_from_checkpoint = 10.8 s, of which file read 0.56 s, load_state_dict
0.19 s, construction 10.0 s - almost all of it random init of weights that strict=True
load_state_dict then overwrites. Skipping it is safe for the weights, but construction also
advances the CPU RNG (torch, and possibly python/numpy), and the dataloader base seed drawn
afterwards drives the featurizer's reference-conformer augmentation. So the RNG must end up
exactly where a real construction leaves it.

The post-construction RNG state is a deterministic function of the pre-construction state and
the code. We record it once from a real construction, keyed by (pre-state hash, boltz commit,
checkpoint sha256, constructor kwargs), and on later loads construct with init as a no-op and
restore the recorded state. Cache miss -> real construction (slow, still exact) + record.

Enable: BOLTZ_FAST_LOAD=1 (via boltz_nw0_exact / serve_batch).
"""
import os
import contextlib
import hashlib
import pickle
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(os.environ.get("BOLTZ2_HOME", Path.home() / "boltz2-dev")) / "cache" / "rng_after_construct"
STATS = {"fast": 0, "recorded": 0}

_RANDOM_INPLACE = ["uniform_", "normal_", "log_normal_", "exponential_", "geometric_",
                   "cauchy_", "random_", "bernoulli_"]
_INIT_FNS = ["uniform_", "normal_", "trunc_normal_", "xavier_uniform_", "xavier_normal_",
             "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_", "constant_",
             "zeros_", "ones_", "eye_", "dirac_"]


def _snapshot():
    return (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())


def _restore(s):
    torch.set_rng_state(s[0])
    random.setstate(s[1])
    np.random.set_state(s[2])


def _hash_state(s):
    h = hashlib.sha256()
    h.update(s[0].numpy().tobytes())
    h.update(pickle.dumps(s[1]))
    h.update(pickle.dumps((s[2][0], s[2][1].tobytes(), s[2][2:])))
    return h.hexdigest()


_SHA_CACHE = {}


def _file_sha(path):
    path = str(path)
    if path not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _SHA_CACHE[path] = h.hexdigest()
    return _SHA_CACHE[path]


def _boltz_commit():
    import boltz
    src = Path(boltz.__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


@contextlib.contextmanager
def _no_init():
    """Make every random/explicit init a no-op while the model is constructed."""
    saved = []
    noop = lambda t, *a, **k: t  # noqa: E731
    for name in _RANDOM_INPLACE:
        saved.append((torch.Tensor, name, getattr(torch.Tensor, name)))
        setattr(torch.Tensor, name, noop)
    for name in _INIT_FNS:
        if hasattr(torch.nn.init, name):
            saved.append((torch.nn.init, name, getattr(torch.nn.init, name)))
            setattr(torch.nn.init, name, noop)
    # Boltz's own helpers (boltz.model.layers.initialize). trunc_normal_init_ samples with
    # scipy.stats.truncnorm.rvs on numpy's global RNG: 8.2 s of the 10.0 s construction.
    # All callers use `init.<fn>(...)` module attribute lookups, so patching here covers them.
    import boltz.model.layers.initialize as binit
    for name in [n for n in vars(binit) if n.endswith("_") and not n.startswith("_")
                 and callable(getattr(binit, n))]:
        saved.append((binit, name, getattr(binit, name)))
        setattr(binit, name, noop)
    try:
        yield
    finally:
        for obj, name, fn in reversed(saved):
            setattr(obj, name, fn)


def install():
    from boltz.model.models.boltz2 import Boltz2

    orig = Boltz2.load_from_checkpoint
    commit = _boltz_commit()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load(checkpoint_path, *a, **k):
        if commit.endswith("+dirty") or commit == "unknown":
            return orig(checkpoint_path, *a, **k)  # never trust a cache for unidentified code
        before = _snapshot()
        kw = repr(sorted((key, repr(v)) for key, v in k.items()))
        key = hashlib.sha256("|".join([_hash_state(before), commit, _file_sha(checkpoint_path),
                                       repr(a), kw]).encode()).hexdigest()[:32]
        rec = CACHE_DIR / f"{key}.pt"
        if rec.exists():
            with _no_init():
                model = orig(checkpoint_path, *a, **k)
            _restore(torch.load(rec, weights_only=False))
            STATS["fast"] += 1
            return model
        model = orig(checkpoint_path, *a, **k)
        torch.save(_snapshot(), rec)
        STATS["recorded"] += 1
        return model

    Boltz2.load_from_checkpoint = staticmethod(load)
    print(f"[fast_load] installed (boltz commit {commit[:12]})", flush=True)
