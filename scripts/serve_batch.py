"""Run many inputs through the unmodified `boltz predict` in ONE process (Phase 2, item 1),
reproducing fresh-process outputs bit-for-bit.

Pays `import boltz` and the checkpoint load once instead of per job. Each input still goes
through upstream's own predict(), which calls seed_everything(seed) on every call.

Exactness: building the model consumes a fixed amount of the CPU RNG stream (random weight init
that the checkpoint then overwrites). The dataloader draws its base seed afterwards, and that seed
drives the featurizer's reference-conformer augmentation. So a cache hit must leave the RNG where
a real load would have. On the first load we record the RNG state just before and just after it;
on a cache hit we verify the state before matches the recorded one, then jump to the recorded
after-state. If the before-state does not match, we do a real load instead (slower, still exact).

Featurization runs in the main process via boltz_nw0_exact (item 2), which emulates worker RNG.

Usage: serve_batch.py <out_root> <times.tsv> <yaml> [<yaml> ...] -- <common boltz predict args>
"""
import json
import os
import random
import sys
import time
import traceback

T0 = time.perf_counter()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boltz_nw0_exact  # noqa: E402  (installs exact main-process featurization)
import boltz.main as bm  # noqa: E402
from boltz.model.models.boltz2 import Boltz2  # noqa: E402

if os.environ.get("BOLTZ_FAST_LOAD") == "1":  # must wrap the loader before we capture it
    import fast_load  # noqa: E402
    fast_load.install()

T_IMPORT = time.perf_counter() - T0


def rng_snapshot():
    return (torch.get_rng_state(), random.getstate(), np.random.get_state())


def rng_equal(a, b):
    return (torch.equal(a[0], b[0]) and a[1] == b[1]
            and a[2][0] == b[2][0] and np.array_equal(a[2][1], b[2][1]) and a[2][2:] == b[2][2:])


def rng_restore(s):
    torch.set_rng_state(s[0])
    random.setstate(s[1])
    np.random.set_state(s[2])


_orig_load = Boltz2.load_from_checkpoint
_cache = {}
STATS = {"real_loads": 0, "cache_hits": 0, "fallbacks": 0}


def cached_load(*args, **kwargs):
    key = repr((args, sorted(kwargs.items())))
    before = rng_snapshot()
    hit = _cache.get(key)
    if hit is not None and rng_equal(before, hit["before"]):
        rng_restore(hit["after"])
        STATS["cache_hits"] += 1
        return hit["model"]
    if hit is not None:
        STATS["fallbacks"] += 1
    model = _orig_load(*args, **kwargs)
    STATS["real_loads"] += 1
    _cache[key] = {"model": model, "before": before, "after": rng_snapshot()}
    return model


Boltz2.load_from_checkpoint = staticmethod(cached_load)


# --- pristine preprocessing ------------------------------------------------------------------
# Ligand conformers come from RDKit ETKDG with randomSeed=-1, which draws from RDKit's internal
# generator: identical at the first embedding of a fresh process, then advancing, and not
# resettable. So each job's preprocessing (process_inputs) runs in a child forked from a
# template that has never embedded anything. The template is forked at startup, before this
# process touches Metal. The server's own process_inputs then finds the records and skips.

def _preprocess_only(job):
    """Mirror predict() up to and including process_inputs, CPU only.

    Must not touch MPS/Objective-C: this runs in a fork() child, and macOS kills a forked child
    that initializes ObjC classes (MPSGraph) the parent was mid-initializing. So: no
    predict.main (accelerator probing) and no seed_everything (torch.manual_seed seeds MPS);
    seed the CPU generators the same way instead.
    """
    from pathlib import Path

    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    cache = Path(job["cache"]).expanduser()
    data = Path(job["yaml"]).expanduser()
    out_dir = Path(job["out_root"]).expanduser() / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bm.download_boltz2(cache)  # existence check only; files are present
    bm.process_inputs(
        data=bm.check_inputs(data), out_dir=out_dir, ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols", use_msa_server=False,
        msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy",
        boltz2=True, preprocessing_threads=1, max_msa_seqs=8192)


def _template_loop(req_r, resp_w):
    with os.fdopen(req_r, "r") as req, os.fdopen(resp_w, "w") as resp:
        for line in req:
            job = json.loads(line)
            pid = os.fork()
            if pid == 0:  # grandchild: preprocess one job, never reaches the model
                code = 1
                try:
                    _preprocess_only(job)
                    code = 0
                except BaseException:  # noqa: BLE001
                    traceback.print_exc()
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(code)
            _, status = os.waitpid(pid, 0)
            resp.write(f"{os.waitstatus_to_exitcode(status)}\n")
            resp.flush()
    os._exit(0)


def start_template():
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(req_w)
        os.close(resp_r)
        _template_loop(req_r, resp_w)
    os.close(req_r)
    os.close(resp_w)
    return pid, os.fdopen(req_w, "w"), os.fdopen(resp_r, "r")


def request_preprocess(template, job):
    """Ask the pristine template to preprocess `job`; returns immediately (runs concurrently)."""
    _, req, _ = template
    req.write(json.dumps(job) + "\n")
    req.flush()


def collect_preprocess(template, job):
    """Block until the oldest outstanding preprocessing request has finished."""
    code = int(template[2].readline())
    if code != 0:
        raise RuntimeError(f"pristine preprocessing failed (exit {code}) for {job['yaml']}")


# --- writer overlap (BOLTZ_OVERLAP=1) ----------------------------------------------------------
# The output writer (mmCIF via modelcif/ihm, compressed npz, json) is CPU/disk work on the
# critical path. With overlap on, predictions are copied to the CPU synchronously and the writing
# runs on one background thread while the next job uses the GPU. Same data is written.

WRITE_FUTURES = []


def _to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x


def enable_writer_overlap():
    from concurrent.futures import ThreadPoolExecutor

    from boltz.data.write.writer import BoltzWriter

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boltz-writer")
    orig = BoltzWriter.write_on_batch_end

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch,
                           batch_idx, dataloader_idx):
        pred = _to_cpu(prediction)
        small_batch = {"record": batch["record"]}
        WRITE_FUTURES.append(pool.submit(orig, self, trainer, pl_module, pred, batch_indices,
                                         small_batch, batch_idx, dataloader_idx))

    BoltzWriter.write_on_batch_end = write_on_batch_end
    return pool


def drain_writes():
    for fut in WRITE_FUTURES:
        fut.result()  # re-raises any writer exception
    WRITE_FUTURES.clear()


def main():
    out_root, times_path = sys.argv[1], sys.argv[2]
    sep = sys.argv.index("--")
    yamls, common = sys.argv[3:sep], sys.argv[sep + 1:]
    if "--num_workers" in common:
        sys.exit("do not pass --num_workers; featurization runs in-process (forced 0)")
    boltz_nw0_exact.apply_precision_override()  # BOLTZ_PRECISION env, item 3; no-op if unset
    boltz_nw0_exact.apply_exact_patches()  # BOLTZ_EXACT_PATCHES=1, item 4; no-op if unset
    overlap = os.environ.get("BOLTZ_OVERLAP") == "1"
    template = start_template()  # before anything touches Metal
    if overlap:
        enable_writer_overlap()
    seed = int(common[common.index("--seed") + 1])
    cache = common[common.index("--cache") + 1]
    jobs = [{"yaml": y, "out_root": out_root, "seed": seed, "cache": cache} for y in yamls]
    with open(times_path, "w") as f:
        f.write(f"input\twall_s\tnote (overlap={overlap})\n")
        f.write(f"(process startup + import)\t{T_IMPORT:.2f}\t\n")
        if overlap and jobs:
            request_preprocess(template, jobs[0])
        for k, y in enumerate(yamls):
            name = y.rsplit("/", 1)[-1].removesuffix(".yaml")
            loads_before = STATS["real_loads"]
            t = time.perf_counter()
            argv = [y, "--out_dir", out_root, *common, "--num_workers", "0"]
            if not overlap:
                request_preprocess(template, jobs[k])
            collect_preprocess(template, jobs[k])
            if overlap and k + 1 < len(jobs):
                request_preprocess(template, jobs[k + 1])  # runs while job k uses the GPU
            bm.predict.main(args=argv, standalone_mode=False)
            dt = time.perf_counter() - t
            note = "real checkpoint load" if STATS["real_loads"] > loads_before else "cached model"
            f.write(f"{name}\t{dt:.2f}\t{note}\n")
            f.flush()
            print(f"{name} {dt:.2f}s {note}", flush=True)
        drain_writes()  # no-op without overlap; the set total includes the last writes
        total = time.perf_counter() - T0
        f.write(f"(set total, whole process)\t{total:.2f}\t{STATS}\n")
    template[1].close()  # template exits on EOF
    os.waitpid(template[0], 0)
    print(f"set total {total:.2f}s {STATS}")


if __name__ == "__main__":
    main()
