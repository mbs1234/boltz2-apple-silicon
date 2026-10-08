"""Stage-level wall-clock profile of one stock `boltz predict` invocation on MPS.

Runs the unmodified CLI in-process, after wrapping a fixed set of methods with timers that
synchronize MPS on entry and exit (so GPU work is attributed to the stage that queued it).
The synchronizations make the total slightly slower than a plain run; use this for attribution,
and the plain driver (run_baseline.sh) for wall-clock claims.

Usage: profile_stages.py <out_json> -- <boltz predict args...>
"""
import json
import sys
import time
from collections import defaultdict
from functools import wraps

T0 = time.perf_counter()
import torch  # noqa: E402

TIMES = defaultdict(float)
CALLS = defaultdict(int)
STACK = []


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(label, fn):
    @wraps(fn)
    def inner(*a, **k):
        sync()
        STACK.append(label)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sync()
            dt = time.perf_counter() - t
            STACK.pop()
            TIMES[label] += dt
            CALLS[label] += 1
            # attribute exclusive time: subtract from the enclosing stage
            if STACK:
                TIMES[STACK[-1] + " (self)"] -= dt
    return inner


def patch_obj_method(obj, name, label):
    setattr(obj, name, timed(label, getattr(obj, name)))


def main():
    out_json = sys.argv[1]
    assert sys.argv[2] == "--"
    argv = sys.argv[3:]

    t = time.perf_counter()
    import boltz.main as bm
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.write.writer import BoltzWriter
    TIMES["import boltz"] = time.perf_counter() - t + (t - T0)

    bm.process_inputs = timed("process_inputs (featurize/preprocess)", bm.process_inputs)
    # Lightning wraps this in its own descriptor; wrap the class-bound callable instead.
    Boltz2.load_from_checkpoint = staticmethod(
        timed("load_from_checkpoint", Boltz2.load_from_checkpoint))
    BoltzWriter.write_on_batch_end = timed("writer", BoltzWriter.write_on_batch_end)

    # Attribute time spent inside Lightning but outside the model: trainer.predict as a whole,
    # moving the model to the device, and waiting on the dataloader (worker spawn + featurize).
    from pytorch_lightning import Trainer
    from torch.utils.data import dataloader as tdl
    Trainer.predict = timed("trainer.predict (total)", Trainer.predict)
    Boltz2.to = timed("model.to(device)", Boltz2.to)
    for cls in (tdl._SingleProcessDataLoaderIter, tdl._MultiProcessingDataLoaderIter):
        cls.__next__ = timed("dataloader next()", cls.__next__)
        cls.__init__ = timed("dataloader iter init (worker spawn)", cls.__init__)

    orig_predict_step = Boltz2.predict_step
    patched = {"done": False}

    def predict_step(self, *a, **k):
        if not patched["done"]:
            for attr, label in [("input_embedder", "input_embedder"),
                                ("msa_module", "msa_module"),
                                ("pairformer_module", "pairformer"),
                                ("distogram_module", "distogram"),
                                ("confidence_module", "confidence")]:
                m = getattr(self, attr, None)
                if m is not None:
                    patch_obj_method(m, "forward", label)
            sm = self.structure_module
            patch_obj_method(sm, "sample", "diffusion sample (total)")
            patch_obj_method(sm.score_model, "forward", "diffusion score_model")
            patched["done"] = True
        return orig_predict_step(self, *a, **k)

    Boltz2.predict_step = timed("predict_step (total)", predict_step)

    t = time.perf_counter()
    try:
        bm.predict.main(args=argv, standalone_mode=False)
    finally:
        TIMES["cli predict (total)"] = time.perf_counter() - t
        TIMES["process (total)"] = time.perf_counter() - T0
        rec = {"times_s": dict(TIMES), "calls": dict(CALLS), "argv": argv}
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=1)
        for k in sorted(TIMES, key=lambda k: -TIMES[k]):
            print(f"{TIMES[k]:9.2f} s  x{CALLS.get(k, '')!s:>4}  {k}")


if __name__ == "__main__":
    main()
