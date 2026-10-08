"""Capture real inputs + weights of one token-level diffusion AttentionPairBias call.

Hooks AttentionPairBias.forward; on the 50th call made with compute_pair_bias=False and
multiplicity>1 (token transformer inside diffusion, mid-trajectory), saves inputs, weights and
the reference output, then stops the job.
Usage: capture_apb.py <out.pt> -- <boltz predict args...>   (no --num_workers)
"""
import sys

import torch

import boltz_nw0_exact  # noqa: F401
from boltz.model.layers.attentionv2 import AttentionPairBias


class _Done(Exception):
    pass


def main():
    out = sys.argv[1]
    argv = sys.argv[sys.argv.index("--") + 1:]
    import boltz.main as bm

    orig = AttentionPairBias.forward
    n = {"k": 0}

    def fwd(self, s, z, mask, k_in, multiplicity=1):
        o = orig(self, s, z, mask, k_in, multiplicity)
        if not self.compute_pair_bias and multiplicity > 1:
            n["k"] += 1
            if n["k"] == 50:
                torch.save({"s": s.cpu(), "z": z.cpu(), "z_stride": z.stride(),
                            "mask": mask.cpu(), "k_in": k_in.cpu(), "multiplicity": multiplicity,
                            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
                            "c_s": self.c_s, "num_heads": self.num_heads, "inf": self.inf,
                            "out": o.cpu()}, out)
                print(f"captured: s {tuple(s.shape)} z {tuple(z.shape)} stride {z.stride()} "
                      f"mask {tuple(mask.shape)} mult {multiplicity}", flush=True)
                raise _Done
        return o

    AttentionPairBias.forward = fwd
    try:
        bm.predict.main(args=[*argv, "--num_workers", "0"], standalone_mode=False)
    except _Done:
        pass


if __name__ == "__main__":
    main()
