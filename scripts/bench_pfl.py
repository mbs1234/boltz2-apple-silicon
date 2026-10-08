"""Microbenchmark a captured PairformerLayer: upstream vs 'skip x1.0 dropout multiply in eval'.

The rewrite still calls get_dropout_mask (so the MPS RNG advances exactly as upstream: in eval
it draws torch.rand and returns all-ones), and only skips `dropout * out`, which is x*1.0.
Both run with exact_patches applied. Outputs must be torch.equal, and the MPS RNG state after
the call must be equal too.
Usage: bench_pfl.py <cap_PairformerLayer.pt>
"""
import sys
import time

import torch

import exact_patches
from boltz.model.layers.dropout import get_dropout_mask
from boltz.model.layers.pairformer import PairformerLayer

exact_patches.apply()
dev = "mps"
c = torch.load(sys.argv[1], weights_only=False)
args = [a["__tensor__"].to(dev) if isinstance(a, dict) else a for a in c["args"]]
m = None
for v2 in (True, False):
    try:
        cand = PairformerLayer(token_s=args[0].shape[-1], token_z=args[1].shape[-1],
                               num_heads=c["init"]["num_heads"], dropout=c["init"]["dropout"],
                               post_layer_norm=c["init"]["post_layer_norm"], v2=v2)
        cand.load_state_dict(c["state_dict"])
        m = cand.to(dev).eval()
        print(f"v2={v2}")
        break
    except RuntimeError:
        continue


def skip_mul(self, s, z, mask, pair_mask, chunk_size_tri_attn=None, use_kernels=False,
             use_cuequiv_mul=False, use_cuequiv_attn=False):
    def add(z, which, out, columnwise=False):
        d = get_dropout_mask(self.dropout, z, self.training, columnwise=columnwise)
        return z + out if not self.training else z + d * out

    z = add(z, 0, self.tri_mul_out(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels))
    z = add(z, 1, self.tri_mul_in(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels))
    z = add(z, 2, self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size_tri_attn,
                                     use_kernels=use_cuequiv_attn or use_kernels))
    # upstream draws the columnwise mask from the z *before* tri_att_end
    d = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
    out = self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size_tri_attn,
                           use_kernels=use_cuequiv_attn or use_kernels)
    z = z + out if not self.training else z + d * out
    z = z + self.transition_z(z)
    with torch.autocast("cuda", enabled=False):
        s_normed = self.pre_norm_s(s.float())
        s = s.float() + self.attention(s=s_normed, z=z.float(), mask=mask.float(), k_in=s_normed)
        s = s + self.transition_s(s)
        s = self.s_post_norm(s)
    return s, z


def run(fn, n=20):
    torch.mps.manual_seed(0)
    with torch.no_grad():
        for _ in range(2):
            fn()
        torch.mps.synchronize()
        torch.mps.manual_seed(0)
        t = time.perf_counter()
        for _ in range(n):
            o = fn()
        torch.mps.synchronize()
        dt = (time.perf_counter() - t) / n * 1e3
    return dt, o, torch.mps.get_rng_state()


t0, (s0, z0), r0 = run(lambda: m(*args))
t1, (s1, z1), r1 = run(lambda: skip_mul(m, *args))
print(f"upstream  {t0:7.2f} ms/layer")
print(f"skip-mul  {t1:7.2f} ms/layer  {100 * (t1 - t0) / t0:+6.1f}%   "
      f"s identical {torch.equal(s0, s1)}  z identical {torch.equal(z0, z1)}  "
      f"MPS RNG state identical {torch.equal(r0, r1)}")
