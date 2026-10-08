# Boltz-2 on Apple Silicon: exact speedups for protein–ligand co-folding

Launchers, patches and a validation harness that make [Boltz-2](https://github.com/jwohlwend/boltz)
co-folding **27–38% faster on Apple Silicon** while producing **bit-identical output** — same
structures, same confidence arrays, same bits.

Also included: two upstream bug fixes, and one finding that a widely assumed optimization does
nothing on this platform.

Measured on Mac Studio M1 Ultra (64-core GPU, 128 GB unified memory), macOS 27.0.1, torch 2.14.1,
on a fixed set of 9 kinase-domain + inhibitor complexes (~300 residues), 5 diffusion samples each.

## Results

| Mode | Wall clock, 9 complexes | vs. stock |
|---|---:|---:|
| Stock `boltz predict`, one process per job | 946.6 s | — |
| **Single-job mode** (`boltz_nw0_exact.py`) | **682.8 s** | **−27.9%** |
| **Batch mode** (`serve_batch.py`) | **582.9 s** | **−38.4%** |

Every accepted change reproduces the reference archive **bit-for-bit**: 45/45 structures and
135/135 pLDDT/PAE/PDE arrays, maximum ligand pocket RMSD 0.000 Å, maximum confidence delta 0.0000.

Re-verified independently on two further M1 Ultra machines: single-job 677 s on both, batch
570 s and 571 s, each 45/45 and 135/135 bit-identical.

## The rule this work is built on

An optimization that changes a pose by 2 Å looks exactly like a speedup. So the acceptance bar
here is not "close enough" — it is bit-identity against a pinned reference archive, checked on
every structure and every confidence array. A change that cannot clear that bar is reported as
rejected, not tuned until it passes.

That bar is what makes the rejected list below as useful as the accepted one.

## What made it faster

### 1. Dataloader workers removed, with the worker RNG reproduced exactly (−15.0%)

`--num_workers 0` is the obvious win: it removes process spawn and IPC per job. It also silently
changes the output, because the featurizer's reference-conformer roto-translation draws from the
**torch global CPU RNG**, whose state differs between a worker process and the main process.
Setting the flag alone gave 0/45 bit-identical structures.

`scripts/boltz_nw0_exact.py` removes the workers *and* emulates the worker-side RNG stream exactly,
restoring bit-identity. The speed comes from the removed workers; the correctness comes from the
emulation.

### 2. Four memory-layout patches (−5.0%, then −2.1%)

In `scripts/exact_patches.py`, enabled with `BOLTZ_EXACT_PATCHES=1`. None changes the arithmetic —
same float operations, different layout:

| Patch | Change | Microbenchmark |
|---|---|---|
| P1 | Diffusion attention-pair-bias hoisted out of the sample loop and broadcast | −15% per call |
| P2 | Triangle-attention ending-node path made contiguous | −23% per call |
| P3 | Pairformer: keep the eval-mode dropout RNG draw, skip the ×1.0 multiply | −3.6% per layer |
| P4 | Outer-product mean: mask count as `maskᵀ @ mask` | −20% per call, avoids a ~3.2 GB temporary |

P3 is the subtle one. The dropout multiply in eval mode is a no-op numerically, but *removing the
RNG draw* would shift every subsequent random number. The patch keeps the draw and drops only the
multiply.

### 3. `fast_load`: skip initialization the checkpoint overwrites (86.9 s → 77.3 s per single job)

Model construction spends 8.2 of 10.0 seconds in `trunc_normal_init_`, which calls SciPy's
`truncnorm` to fill weights that `load_state_dict(strict=True)` then overwrites completely.
`scripts/fast_load.py` skips that initialization and restores the recorded post-construction
CPU/numpy RNG state, so the downstream stream is unchanged.

The cache is keyed on the pre-state hash, the boltz commit, the checkpoint sha256 and the
constructor kwargs, and it refuses to engage on a dirty or non-git checkout — if you modify the
source, it turns itself off and runs the slow path rather than silently serving a stale state.

### 4. Overlapped I/O for batches (−6.6%)

`BOLTZ_OVERLAP=1` prefetches the next job's preprocessing and writes outputs on a background
thread. Still one GPU process throughout. This is I/O and CPU work moved off the critical path,
not concurrency — see the hardware note below.

## What did not work, and why

This is the more informative half.

| Attempt | Result | Why it failed |
|---|---|---|
| Persistent process with a cached model | 817.2 s (−13.7%) but **5/45** identical | Skipping construction changes the dataloader base seed; RDKit's ETKDG generator also advances between jobs |
| Plain `--num_workers 0` | 802.1 s (−15.3%) but **0/45** identical | Featurizer RNG state differs between worker and main process |
| Persistent process + RNG restore | 676.3 s (−28.6%) but **20/45** identical | Apo inputs and the first ligand job match; later ligand jobs drift on RDKit's **unseeded** ETKDG generator |
| `--precision 32` | No change | **See below** |

The persistent-process family is what motivated `serve_batch.py`, which gets the same speed
*and* bit-identity by running each job's preprocessing in a pristine forked child.

Also declined after measurement, for being too small or needing an unproven correctness argument:
template-skip (~0.8% per job), atom-transformer copy hoisting (≤0.19 s per job), a
natural-layout PWA einsum (+0.0%, bit-identical but no gain), `weighted_rigid_align` (0.7% of
model time), and triangle-attention chunking (inactive below 384 tokens).

## Finding: bf16 mixed precision was never active on MPS

Lightning 2.5.0 logs `Using bfloat16 Automatic Mixed Precision (AMP)` and then runs fp32 on Apple
MPS. It builds the autocast context for device `cuda` whenever the accelerator is not `cpu`
(`accelerator_connector.py:523`), and a `cuda` autocast context does not affect MPS tensors.

So the precision flag was a no-op: the measured −0.9% for forcing fp32 is within noise, because
inference was already running fp32. Patch `0001` makes MPS select fp32 explicitly and say so.

If you have been comparing bf16 against fp32 timings on Apple Silicon, you were likely comparing
fp32 against fp32.

## Two upstream bug fixes

`patches/0001-mps-precision-and-subsample-msa-flag.diff`
- MPS precision is selected explicitly as fp32 and logged truthfully (above).
- `--subsample_msa` becomes a working `--subsample_msa/--no_subsample_msa` switch with accurate
  help text. **Default stays off.**
- Bit-identical to stock: 45/45 structures, 135/135 arrays. No speed change.

`patches/0002-augment-last-reference-conformer-group.diff`
- `for i in range(torch.max(ref_space_uid))` never augments the last conformer group, because
  `ref_space_uid` runs `0..max` inclusive. In a protein + ligand input the last group is **always
  the ligand**, so the ligand's reference conformer was never randomly roto-translated.
- Fixed to `range(int(torch.max(ref_space_uid)) + 1)`.

The second fix changes inputs and the RNG stream, so it is a **different instrument** and was
validated distributionally rather than by bit-identity: 15 samples per input (seeds 0, 1, 2)
against 15 stock samples. Every confidence mean lies inside stock's observed range; the largest
shift is 1.35 stock SD. Cross-set pose and Cα spreads equal stock's own within-set spreads — that
is, outputs from the fixed code are as close to stock outputs as stock outputs are to each other.

## Hardware notes for Apple Silicon

Measured, not quoted from a spec sheet. These shaped every decision above:

- **M1 has no GPU matrix-multiply units.** A100-era precision tricks are neutral or negative here.
  The wins available are bandwidth, memory layout, avoided recompute and I/O.
- **Concurrency buys nothing.** The same 96-prediction workload across 1/2/4/6 concurrent
  processes took 1266/1184/1198/1246 s — a 7% spread, within noise. Metal serializes GPU work, so
  low CPU utilization is not recoverable headroom. Run one GPU process at a time.
- **`fork()` after touching Metal crashes.** `serve_batch.py` forks its preprocessing template
  before any GPU use. Never call MPS in a forked child.
- **Dataloader workers use spawn on macOS**, so every entry point needs an
  `if __name__ == "__main__":` guard or it re-imports and recurses.

## Contents

```
scripts/boltz_nw0_exact.py   single-job launcher: no workers + exact RNG emulation
scripts/serve_batch.py       batch launcher: persistent process, pristine forked preprocessing
scripts/exact_patches.py     P1-P4 layout patches, enabled by env var
scripts/fast_load.py         skip initialization the checkpoint overwrites
scripts/validate_node.sh     acceptance harness: gates G0-G7
scripts/mps_probe.py         MPS-vs-CPU parity probe (GEMM, attention, reductions)
scripts/compare.py           paired bit-identity comparison
scripts/compare_sets.py      distributional comparison for different-instrument changes
scripts/bench_*.py           per-module microbenchmarks on captured inputs
patches/                     the two upstream fixes as plain diffs
provenance/weights.json      pinned weights revision and sha256 checksums
```

The launchers are wrappers around the unmodified CLI, enabled by environment variables. Upstream
source files are not edited.

## Usage

Apply the patches to a clean checkout of boltz at `b1ebfc4`, install editable, then:

```bash
# one co-fold
BOLTZ_EXACT_PATCHES=1 BOLTZ_FAST_LOAD=1 python scripts/boltz_nw0_exact.py \
  target.yaml --out_dir OUT --cache ./cache \
  --diffusion_samples 5 --accelerator gpu --no_kernels --seed 0

# a series, one process
BOLTZ_EXACT_PATCHES=1 BOLTZ_OVERLAP=1 BOLTZ_FAST_LOAD=1 python scripts/serve_batch.py \
  OUT OUT/times.tsv a.yaml b.yaml c.yaml -- --cache ./cache \
  --diffusion_samples 5 --accelerator gpu --no_kernels --seed 0
```

`fast_load` requires a clean git checkout. A dirty tree disables it — slower, still correct.

## Validating on your own machine

`scripts/validate_node.sh` runs gates G0–G7: environment fingerprint, code commit, package
versions, MPS parity, weight checksums, then the fixed input set three ways against a reference
archive.

**You will need to build your own reference archive first.** The validation set used here is nine
kinase + inhibitor complexes, two of which are unpublished compounds and are therefore not
included. Generate a baseline with stock boltz on your own inputs, then check the optimized modes
against it. The harness and the comparison scripts are general; only the reference data is absent.

A node is numerically different — and the harness will say so — if the macOS build, torch version
or GPU core count differs from the one the archive was built on.

## Scope

Validated: single protein chain + small-molecule ligand, up to 384 tokens, 5 diffusion samples,
full MSA, `--subsample_msa` off.

**Not validated.** Treat as untested: the affinity module, complexes over 384 tokens (triangle-
attention chunking switches on there and was never exercised), multi-chain proteins, templates,
constraints and pockets, CCD-code ligands, covalent ligands, nucleic acids.

## Licence and attribution

Boltz-2 is by Jeremy Wohlwend, Gabriele Corso, Saro Passaro and contributors, released under the
MIT License — see `LICENSE.boltz`. The diffs in `patches/` are modifications of that MIT-licensed
source and carry the same licence.

The scripts in this repository are released under the MIT License — see `LICENSE`.

Model weights are not redistributed here. `provenance/weights.json` pins the revision and
checksums of the public `boltz-community/boltz-2` release used for all measurements.
