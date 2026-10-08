"""Microbenchmark of one captured diffusion AttentionPairBias call: original vs exact rewrites.

Every variant must be bit-identical (torch.equal) to the original's output on the same device.
Usage: bench_apb.py <apb_capture.pt>
"""
import sys
import time

import torch

from boltz.model.layers.attentionv2 import AttentionPairBias

dev = "mps"
cap = torch.load(sys.argv[1], weights_only=False)
m = AttentionPairBias(c_s=cap["c_s"], num_heads=cap["num_heads"], inf=cap["inf"],
                      compute_pair_bias=False)
m.load_state_dict(cap["state_dict"])
m = m.to(dev).eval()
s, mask, k_in, mult = (cap["s"].to(dev), cap["mask"].to(dev), cap["k_in"].to(dev),
                       cap["multiplicity"])
# rebuild z as the same strided slice the model passes (last dim of a 24-layer interleaved bias)
L = cap["z_stride"][2] // cap["z"].shape[-1]
full = torch.zeros(*cap["z"].shape[:3], cap["z"].shape[-1] * L, device=dev)
li = 7
full.view(*cap["z"].shape[:3], L, -1)[..., li, :] = cap["z"].to(dev)
z = full.view(*cap["z"].shape[:3], L, -1)[..., li, :]
assert z.stride() == tuple(cap["z_stride"]), (z.stride(), cap["z_stride"])


def core(self, s, bias_b1hij, mask, k_in, B_eff, mult):
    """Shared tail of the rewrites: identical op sequence to upstream, bias broadcast over mult."""
    B = s.shape[0]
    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
    g = self.proj_g(s).sigmoid()
    attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
    attn = attn / (self.head_dim**0.5)
    attn = (attn.view(B_eff, mult, *attn.shape[1:]) + bias_b1hij).view(attn.shape)
    attn = attn + (1 - mask[:, None, None].float()) * -self.inf
    attn = attn.softmax(dim=-1)
    o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    o = o.reshape(B, -1, self.c_s)
    return self.proj_o(g * o)


def v_broadcast(self, s, z, mask, k_in, multiplicity=1):
    bias = self.proj_z(z).float()  # Rearrange view, no copy
    return core(self, s, bias[:, None], mask, k_in, z.shape[0], multiplicity)


PRE = {}


def v_hoisted(self, s, z, mask, k_in, multiplicity=1):
    key = (z.data_ptr(), z.shape, z.stride())
    if key not in PRE:  # once per diffusion trajectory in real use
        PRE[key] = self.proj_z(z).float().contiguous()[:, None]
    return core(self, s, PRE[key], mask, k_in, z.shape[0], multiplicity)


def bench(fn, n=50):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            out = fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3, out


with torch.no_grad():
    t0, ref = bench(lambda: m(s, z, mask, k_in, mult))
    print(f"original      {t0:7.2f} ms/call   (reference)")
    print(f"  matches captured model output: {torch.equal(ref.cpu(), cap['out'])}")
    for name, f in (("broadcast", v_broadcast), ("hoisted+bcast", v_hoisted)):
        t, o = bench(lambda f=f: f(m, s, z, mask, k_in, mult))
        print(f"{name:13s} {t:7.2f} ms/call   {100 * (t - t0) / t0:+6.1f}%   "
              f"bit-identical: {torch.equal(o, ref)}")
    # where does the original's time go?
    B = s.shape[0]
    tq, _ = bench(lambda: [m.proj_q(s), m.proj_k(k_in), m.proj_v(k_in), m.proj_g(s)])
    print(f"  of which q/k/v/g projections ~{tq:6.2f} ms")
