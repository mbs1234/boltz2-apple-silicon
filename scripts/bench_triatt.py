"""Microbenchmark captured pairformer TriangleAttention (starting / ending node) calls:
original vs exact rewrites. Every variant must be torch.equal to the original on MPS.
Usage: bench_triatt.py <cap_TriangleAttentionEndingNode.pt> <cap_TriangleAttention.pt>
"""
import sys
import time

import torch

from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.layers.triangular_attention.primitives import permute_final_dims  # noqa: F401

dev = "mps"


def restore(x):
    if isinstance(x, dict) and "__tensor__" in x:
        t = x["__tensor__"].to(dev)
        return t if t.stride() == tuple(x["stride"]) else t.as_strided(t.shape, x["stride"])
    return x


def load(path):
    c = torch.load(path, weights_only=False)
    i = c["init"]
    m = TriangleAttention(c_in=i["c_in"], c_hidden=i["c_hidden"], no_heads=i["no_heads"],
                          starting=i["starting"], inf=i["inf"])
    m.load_state_dict(c["state_dict"])
    m = m.to(dev).eval()
    x = restore(c["args"][0])
    kw = {k: restore(v) for k, v in c["kwargs"].items()}
    return m, x, kw, c["out"]["__tensor__"]


def variant(self, x, mask=None, chunk_size=None, use_kernels=False, contig=False,
            skip_single_chunk=False):
    """Upstream TriangleAttention.forward with two optional, value-preserving changes."""
    from boltz.model.layers.triangular_attention.attention import permute_final_dims as pfd
    if mask is None:
        mask = x.new_ones(x.shape[:-1])
    if not self.starting:
        x = x.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
        if contig:
            x = x.contiguous()
            mask = mask.contiguous()
    x = self.layer_norm(x)
    mask = mask[..., :, None, None, :]
    mask_bias = self.inf * (mask - 1)
    triangle_bias = pfd(self.linear(x), (2, 0, 1)).unsqueeze(-4)
    single = skip_single_chunk and chunk_size is not None and chunk_size >= x.shape[-3]
    if chunk_size is not None and not use_kernels and not single:
        x = self._chunk(x, triangle_bias, mask_bias, mask, chunk_size, use_kernels=use_kernels)
    else:
        x = self.mha(x, x, triangle_bias, mask_bias, mask, use_kernels=use_kernels)
    if not self.starting:
        x = x.transpose(-2, -3)
    return x


def bench(fn, n=30):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        o = fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3, o


with torch.no_grad():
    for path in sys.argv[1:3]:
        m, x, kw, ref_cap = load(path)
        tag = "ending" if not m.starting else "starting"
        t0, ref = bench(lambda: m(x, **kw))
        print(f"[{tag}] original {t0:7.2f} ms  (matches captured: {torch.equal(ref.cpu(), ref_cap)})")
        for name, opts in (("contiguous", {"contig": True}),
                           ("no-chunk", {"skip_single_chunk": True}),
                           ("both", {"contig": True, "skip_single_chunk": True})):
            if name != "no-chunk" and m.starting:
                continue
            t, o = bench(lambda o=opts: variant(m, x, **kw, **o))
            print(f"[{tag}] {name:10s} {t:7.2f} ms  {100 * (t - t0) / t0:+6.1f}%  "
                  f"bit-identical: {torch.equal(o, ref)}  out contiguous: {o.is_contiguous()}")
