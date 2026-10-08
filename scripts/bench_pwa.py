"""Microbenchmark a captured MSA-module PairWeightedAveraging call (unchunked branch).

Variant 'natural layout': keep v as [b, s, j, h, d] and ask einsum for o as [b, s, i, h, d], so
neither the v permute nor the o permute/reshape copy is needed. Unlike the other patches this is
NOT exact by construction (einsum may choose a different bmm arrangement), so torch.equal decides.
Usage: bench_pwa.py <cap_PairWeightedAveraging.pt>
"""
import sys
import time

import torch

from boltz.model.layers.pair_averaging import PairWeightedAveraging

dev = "mps"
c = torch.load(sys.argv[1], weights_only=False)
m_in, z_in, mask_in = (c["args"][i]["__tensor__"].to(dev) for i in range(3))
i = c["init"]
mod = PairWeightedAveraging(c_m=i["c_m"], c_z=i["c_z"], c_h=i["c_h"], num_heads=i["num_heads"],
                            inf=i["inf"])
mod.load_state_dict(c["state_dict"])
mod = mod.to(dev).eval()


def natural(self, m, z, mask):
    m = self.norm_m(m)
    z = self.norm_z(z)
    v = self.proj_m(m).reshape(*m.shape[:3], self.num_heads, self.c_h)   # b s j h d
    b = self.proj_z(z).permute(0, 3, 1, 2)
    b = b + (1 - mask[:, None]) * -self.inf
    w = torch.softmax(b, dim=-1)                                             # b h i j
    g = self.proj_g(m).sigmoid()
    o = torch.einsum("bhij,bsjhd->bsihd", w, v)
    o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
    return self.proj_o(g * o)


def bench(fn, n=8):
    with torch.no_grad():
        fn()
        torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            o = fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3, o


with torch.no_grad():
    t0, ref = bench(lambda: mod(m_in, z_in, mask_in, False))
    print(f"upstream        {t0:8.2f} ms/call  (matches captured: {torch.equal(ref.cpu(), c['out']['__tensor__'])})")
    t1, o1 = bench(lambda: natural(mod, m_in, z_in, mask_in))
    d = (o1 - ref).abs().max().item()
    print(f"natural layout  {t1:8.2f} ms/call  {100 * (t1 - t0) / t0:+6.1f}%  "
          f"bit-identical: {torch.equal(o1, ref)}  max|diff| {d:.3g}")
