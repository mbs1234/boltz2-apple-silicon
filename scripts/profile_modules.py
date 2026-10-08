"""Module-level wall-clock attribution inside one `boltz predict` job on MPS (Phase 2, item 4).

At the first predict_step, every submodule's forward (and AtomDiffusion.sample, and
weighted_rigid_align) is wrapped with a timer that synchronizes MPS on entry and exit.
Time is attributed *exclusively* (a parent's time excludes its timed children) and aggregated
by (top-level block, class name). Synchronizing at every module boundary removes GPU/CPU overlap,
so totals run slower than an unprofiled job; the report states the inflation.

Runs through boltz_nw0_exact (featurization in-process, archive-exact).
Usage: profile_modules.py <out_json> -- <boltz predict args...>   (no --num_workers)
"""
import json
import sys
import time
from collections import defaultdict
from functools import wraps

import torch

import boltz_nw0_exact  # noqa: F401

EXCL = defaultdict(float)
INCL = defaultdict(float)
CALLS = defaultdict(int)
STACK = []  # [label, child_time]


def sync():
    torch.mps.synchronize()


def timed(label, fn):
    @wraps(fn)
    def inner(*a, **k):
        sync()
        STACK.append([label, 0.0])
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sync()
            dt = time.perf_counter() - t
            _, child = STACK.pop()
            EXCL[label] += dt - child
            INCL[label] += dt
            CALLS[label] += 1
            if STACK:
                STACK[-1][1] += dt
    return inner


# Block-level allowlist: leaf modules (Linear, LayerNorm, ...) are deliberately NOT timed, since
# a sync per leaf call (~180k per job) inflated predict_step ~3x and swamped the signal.
BLOCKS = {
    "RelativePositionEncoder",
    "DiffusionModule", "SingleConditioning", "PairwiseConditioning", "DiffusionConditioning",
    "AtomAttentionEncoder", "AtomAttentionDecoder", "DiffusionTransformer", "AtomTransformer",
    "PairformerModule", "PairformerLayer", "TriangleMultiplicationOutgoing",
    "TriangleMultiplicationIncoming", "TriangleAttention", "TriangleAttentionStartingNode",
    "TriangleAttentionEndingNode", "AttentionPairBias", "Transition",
    "MSAModule", "MSALayer", "OuterProductMean", "PairWeightedAveraging",
    "InputEmbedder", "ConfidenceModule", "TemplateV2Module", "TemplateModule", "DistogramModule",
    "ContactConditioning",
}
ALL_MODULES = False  # True reproduces the leaf-level (heavily inflated) profile

TOPS = ["input_embedder", "rel_pos", "contact_conditioning", "msa_module", "pairformer_module",
        "template_module", "diffusion_conditioning",
        "distogram_module", "structure_module", "confidence_module", "bfactor_module"]


def instrument(model):
    inventory = {}
    for top in TOPS:
        sub = getattr(model, top, None)
        if sub is None:
            continue
        counts = defaultdict(int)
        for m in sub.modules():
            cls = type(m).__name__
            counts[cls] += 1
            if ALL_MODULES or cls in BLOCKS:
                m.forward = timed(f"{top}/{cls}", m.forward)
        inventory[top] = dict(counts)
    sm = model.structure_module
    sm.sample = timed("structure_module/AtomDiffusion.sample", sm.sample)
    import boltz.model.modules.diffusionv2 as dv2
    dv2.weighted_rigid_align = timed("structure_module/weighted_rigid_align(fn)",
                                     dv2.weighted_rigid_align)
    return inventory


def main():
    out_json = sys.argv[1]
    argv = sys.argv[sys.argv.index("--") + 1:]
    import boltz.main as bm
    from boltz.model.models.boltz2 import Boltz2

    orig = Boltz2.predict_step
    state = {"inv": None, "t_predict": 0.0}

    def predict_step(self, *a, **k):
        if state["inv"] is None:
            state["inv"] = instrument(self)
        sync()
        t = time.perf_counter()
        out = orig(self, *a, **k)
        sync()
        state["t_predict"] += time.perf_counter() - t
        return out

    Boltz2.predict_step = predict_step
    bm.predict.main(args=[*argv, "--num_workers", "0"], standalone_mode=False)

    tp = state["t_predict"]
    rows = sorted(EXCL.items(), key=lambda kv: -kv[1])
    rec = {"predict_step_s": tp, "attributed_s": sum(EXCL.values()),
           "exclusive_s": dict(EXCL), "inclusive_s": dict(INCL), "calls": dict(CALLS),
           "inventory": state["inv"]}
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"predict_step {tp:.2f}s, attributed {rec['attributed_s']:.2f}s")
    for k, v in rows[:40]:
        print(f"{v:8.2f}s {100 * v / tp:5.1f}%  x{CALLS[k]:<7d} {k}")


if __name__ == "__main__":
    main()
