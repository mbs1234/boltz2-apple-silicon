"""Microbenchmark a captured MSA-module OuterProductMean call (unchunked branch).

Upstream computes num_mask = (mask[:,:,None,:] * mask[:,:,:,None]).sum(1), materializing a
[B, S, N, N] tensor (S = 8192 MSA rows here, ~3.2 GB) to reduce it straight back to [B, N, N].
Rewrite: num_mask = mask^T @ mask. Every term is 0/1 and every partial sum an integer <= S < 2^24,
so all summation orders are exact in fp32 -> bit-identical. Must be torch.equal end to end.
Usage: bench_opm.py <cap_OuterProductMean.pt>
"""
import sys
import time

import torch

from boltz.model.layers.outer_product_mean import OuterProductMean

dev = "mps"
c = torch.load(sys.argv[1], weights_only=False)
m_in = c["args"][0]["__tensor__"].to(dev)
mask_in = c["args"][1]["__tensor__"].to(dev)
sd = c["state_dict"]
mod = OuterProductMean(c_in=m_in.shape[-1], c_hidden=c["init"]["c_hidden"],
                       c_out=sd["proj_o.weight"].shape[0])
mod.load_state_dict(sd)
mod = mod.to(dev).eval()


def matmul_count(self, m, mask, chunk_size=None):
    assert chunk_size is None
    mask = mask.unsqueeze(-1).to(m)
    m = self.norm(m)
    a = self.proj_a(m) * mask
    b = self.proj_b(m) * mask
    mk = mask[..., 0]                                  # [B, S, N]
    num_mask = (mk.transpose(-1, -2) @ mk).clamp(min=1)[..., None]  # [B, N, N, 1]
    z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
    z = z.reshape(*z.shape[:3], -1)
    z = z / num_mask
    return self.proj_o(z.to(m))


def bench(fn, n=10):
    with torch.no_grad():
        fn()
        torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            o = fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3, o


with torch.no_grad():
    # verify the count itself first
    mk = mask_in
    ref_count = (mk[:, :, None, :] * mk[:, :, :, None]).sum(1).clamp(min=1)
    new_count = (mk.transpose(-1, -2) @ mk).clamp(min=1)
    print("num_mask identical:", torch.equal(ref_count, new_count),
          "| max count", int(ref_count.max()))
    t0, ref = bench(lambda: mod(m_in, mask_in, None))
    print(f"upstream      {t0:8.2f} ms/call  (matches captured: {torch.equal(ref.cpu(), c['out']['__tensor__'])})")
    t1, o1 = bench(lambda: matmul_count(mod, m_in, mask_in, None))
    print(f"matmul count  {t1:8.2f} ms/call  {100 * (t1 - t0) / t0:+6.1f}%  bit-identical: {torch.equal(o1, ref)}")
    tc, _ = bench(lambda: (mk[:, :, None, :] * mk[:, :, :, None]).sum(1))
    print(f"  (upstream mask-count step alone: {tc:.2f} ms)")
