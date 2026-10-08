"""Value-preserving performance patches for upstream Boltz-2 on MPS (Phase 2, item 4).

Each patch performs the same floating-point operations on the same values as upstream; only
memory layout / where step-invariant work happens changes. Each was verified bit-identical
(torch.equal) on captured real inputs before being used here, and the full fixed set must still
reproduce the Phase 1 archive 45/45.

  P1  diffusion AttentionPairBias (compute_pair_bias=False, multiplicity>1):
      upstream rebuilds `bias.repeat_interleave(multiplicity)` from a strided per-layer slice at
      every diffusion step. Here the bias is made contiguous once per trajectory and broadcast
      over the samples. Same adds, no per-step copy.        microbench: -15% per call
  P2  TriangleAttentionEndingNode: upstream runs LayerNorm, the bias projection and attention on
      a transposed (strided) view. Here the transposed input is made contiguous first.
                                                              microbench: -23% per call

Enable with `import exact_patches; exact_patches.apply()`.
"""
import inspect

import torch

_BIAS_CACHE = {}


def _takes_flash(fn):
    """boltz-community adds `use_flash_attn` to these forwards; stock 2.2.1 does not."""
    return "use_flash_attn" in inspect.signature(fn).parameters


def _patch_attention_pair_bias():
    from boltz.model.layers.attentionv2 import AttentionPairBias

    orig = AttentionPairBias.forward
    flash = _takes_flash(orig)

    def forward(self, s, z, mask, k_in, multiplicity=1, use_flash_attn=False):
        if self.compute_pair_bias or multiplicity == 1 or use_flash_attn:
            if flash:
                return orig(self, s, z, mask, k_in, multiplicity, use_flash_attn=use_flash_attn)
            return orig(self, s, z, mask, k_in, multiplicity)
        key = (z.data_ptr(), tuple(z.shape), z.stride())
        bias = _BIAS_CACHE.get(key)
        if bias is None:
            bias = self.proj_z(z).float().contiguous()[:, None]  # [B, 1, H, I, J]
            _BIAS_CACHE[key] = bias
        B = s.shape[0]
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        g = self.proj_g(s).sigmoid()
        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5)
            attn = (attn.view(z.shape[0], multiplicity, *attn.shape[1:]) + bias).view(attn.shape)
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, self.c_s)
        return self.proj_o(g * o)

    AttentionPairBias.forward = forward

    # The cached bias is only valid within one diffusion trajectory: clear it around sample().
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    orig_sample = AtomDiffusion.sample

    def sample(self, *a, **k):
        _BIAS_CACHE.clear()
        try:
            return orig_sample(self, *a, **k)
        finally:
            _BIAS_CACHE.clear()

    AtomDiffusion.sample = sample


def _patch_triangle_attention_ending():
    from boltz.model.layers.triangular_attention import attention as ta

    orig = ta.TriangleAttention.forward
    flash = _takes_flash(orig)

    def forward(self, x, mask=None, chunk_size=None, use_kernels=False, use_flash_attn=False):
        if use_flash_attn or self.starting:
            # starting node is unchanged by P2; flash path belongs to the fork, leave it alone
            if flash:
                return orig(self, x, mask, chunk_size, use_kernels, use_flash_attn=use_flash_attn)
            return orig(self, x, mask, chunk_size, use_kernels)
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3).contiguous()
            mask = mask.transpose(-1, -2).contiguous()
        x = self.layer_norm(x)
        mask = mask[..., :, None, None, :]
        mask_bias = self.inf * (mask - 1)
        triangle_bias = ta.permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        extra_chunk = {"use_flash_attn": False} if flash else {}
        extra_mha = {"use_sdpa": False} if flash else {}
        if chunk_size is not None and not use_kernels:
            x = self._chunk(x, triangle_bias, mask_bias, mask, chunk_size,
                            use_kernels=use_kernels, **extra_chunk)
        else:
            x = self.mha(x, x, triangle_bias, mask_bias, mask, use_kernels=use_kernels,
                         **extra_mha)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x

    ta.TriangleAttention.forward = forward


def _patch_pairformer_dropout_mul():
    """P3: in eval, get_dropout_mask returns all-ones but still draws torch.rand on the device,
    advancing the RNG that later drives diffusion noise. Keep the draw (RNG stream unchanged),
    skip only `ones * out` (x*1.0 is exact). microbench: -3.6%/layer, RNG state identical."""
    from boltz.model.layers import pairformer as pf

    orig = pf.PairformerLayer.forward
    flash = _takes_flash(orig)

    def forward(self, s, z, mask, pair_mask, chunk_size_tri_attn=None, use_kernels=False,
                use_cuequiv_mul=False, use_cuequiv_attn=False, use_flash_attn=False):
        if self.training or use_flash_attn:
            extra = {"use_flash_attn": use_flash_attn} if flash else {}
            return orig(self, s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels,
                        use_cuequiv_mul, use_cuequiv_attn, **extra)
        att_extra = {"use_flash_attn": False} if flash else {}
        gdm = pf.get_dropout_mask
        gdm(self.dropout, z, self.training)
        z = z + self.tri_mul_out(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels)
        gdm(self.dropout, z, self.training)
        z = z + self.tri_mul_in(z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels)
        gdm(self.dropout, z, self.training)
        z = z + self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size_tri_attn,
                                   use_kernels=use_cuequiv_attn or use_kernels, **att_extra)
        gdm(self.dropout, z, self.training, columnwise=True)
        z = z + self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size_tri_attn,
                                 use_kernels=use_cuequiv_attn or use_kernels, **att_extra)
        z = z + self.transition_z(z)
        with torch.autocast("cuda", enabled=False):
            s_normed = self.pre_norm_s(s.float())
            s = s.float() + self.attention(s=s_normed, z=z.float(), mask=mask.float(),
                                           k_in=s_normed)
            s = s + self.transition_s(s)
            s = self.s_post_norm(s)
        return s, z

    pf.PairformerLayer.forward = forward


def _patch_outer_product_mean_count():
    """P4: unchunked OuterProductMean built a [B, S, N, N] tensor (S=8192 MSA rows, ~3.2 GB) only
    to sum it over S. mask^T @ mask gives the same counts; all terms are 0/1 and all partial sums
    integers <= S < 2^24, so every summation order is exact in fp32. microbench: -20%/call."""
    from boltz.model.layers.outer_product_mean import OuterProductMean

    orig = OuterProductMean.forward

    def forward(self, m, mask, chunk_size=None):
        if chunk_size is not None and not self.training:
            return orig(self, m, mask, chunk_size)
        mask = mask.unsqueeze(-1).to(m)
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask
        mk = mask[..., 0]
        num_mask = (mk.transpose(-1, -2) @ mk).clamp(min=1)[..., None]
        z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask
        return self.proj_o(z.to(m))

    OuterProductMean.forward = forward


APPLIED = []


def apply():
    if APPLIED:
        return
    _patch_attention_pair_bias()
    _patch_triangle_attention_ending()
    _patch_pairformer_dropout_mul()
    _patch_outer_product_mean_count()
    APPLIED.extend(["P1_apb_bias_hoist", "P2_triatt_end_contiguous",
                    "P3_pairformer_skip_x1_dropout_mul", "P4_opm_matmul_mask_count"])
    print(f"[exact_patches] applied: {APPLIED}", flush=True)
