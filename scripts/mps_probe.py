# Copied verbatim from ~/m1-setup/m1-acceptance-check.sh (section 7 probe body), 2026-10-07.
"""Does this machine's MPS backend exist, and does it agree with the CPU?

Availability is not the question that matters. pytorch/pytorch#187280 reports MPS returning
silently corrupted output on macOS 27 while the CPU path on the same machine is correct, and
pytorch/test-infra#8964 (1 Oct 2026) says upstream MPS CI has no macOS 27 bare-metal coverage at
all. So a co-folding model will return a confident, plausible, correctly-folded kinase whether or
not the numerics underneath it are sound. This probe is the cheap version of that question:
deterministic tensors, both precisions, max abs difference against the CPU as reference.
"""
import json, platform, sys
rec = {"python": sys.version.split()[0], "arch": platform.machine(),
       "macos": platform.mac_ver()[0]}
try:
    import numpy as np
    cfg = ""
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        cfg = buf.getvalue().lower()
    except Exception:
        pass
    rec["numpy"] = np.__version__
    rec["blas_accelerate"] = ("accelerate" in cfg)
except Exception as e:
    rec["numpy_error"] = repr(e)

try:
    import torch
    rec["torch"] = torch.__version__
    rec["mps_built"] = bool(torch.backends.mps.is_built())
    rec["mps_available"] = bool(torch.backends.mps.is_available())
    if rec["mps_available"]:
        torch.manual_seed(0)
        # three shapes that between them cover what a co-folding stack actually does:
        # a big GEMM, a softmax-attention block, and a reduction.
        res = {}
        for name, dtype, tol in (("fp32", torch.float32, 2e-3), ("fp16", torch.float16, 5e-2)):
            a = torch.randn(2048, 2048)
            b = torch.randn(2048, 2048)
            ref = (a.float() @ b.float())
            got = (a.to("mps", dtype) @ b.to("mps", dtype)).float().cpu()
            gemm = (ref - got).abs().max().item() / ref.abs().max().item()

            q = torch.randn(4, 8, 256, 64); k = torch.randn(4, 8, 256, 64); v = torch.randn(4, 8, 256, 64)
            ref_a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            got_a = torch.nn.functional.scaled_dot_product_attention(
                q.to("mps", dtype), k.to("mps", dtype), v.to("mps", dtype)).float().cpu()
            attn = (ref_a - got_a).abs().max().item() / ref_a.abs().max().item()

            x = torch.randn(1_000_000)
            red = abs(x.sum().item() - x.to("mps", dtype).float().sum().item()) / abs(x.sum().item() + 1e-9)
            res[name] = {"gemm_rel": gemm, "attn_rel": attn, "sum_rel": red,
                         "tol": tol, "pass": bool(gemm < tol and attn < tol)}
        rec["parity"] = res
        rec["parity_pass"] = all(v["pass"] for v in res.values())
        import time
        m1 = torch.randn(4096, 4096, device="mps")
        m2 = torch.randn(4096, 4096, device="mps")
        _ = m1 @ m2; torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = m1 @ m2
        torch.mps.synchronize()
        rec["mps_gemm_tflops"] = round(20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12, 2)
    else:
        rec["parity_pass"] = None
except Exception as e:
    rec["torch_error"] = repr(e)
print(json.dumps(rec, indent=1))
