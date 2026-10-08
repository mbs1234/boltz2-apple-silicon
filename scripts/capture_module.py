"""Capture real inputs, weights and output of the Nth forward call of a module class.

Usage: capture_module.py <out.pt> <ClassName> <N> [<top_attr>] -- <boltz predict args...>
  top_attr (optional): only count calls on modules under this Boltz2 attribute,
  e.g. pairformer_module. Stops the job after capturing.
"""
import sys

import torch

import boltz_nw0_exact  # noqa: F401


class _Done(Exception):
    pass


def to_cpu(x):
    if torch.is_tensor(x):
        return {"__tensor__": x.detach().cpu(), "stride": x.stride()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x


def main():
    out, cls_name, nth = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sep = sys.argv.index("--")
    top = sys.argv[4] if sep > 4 else None
    argv = sys.argv[sep + 1:]
    import boltz.main as bm
    from boltz.model.models.boltz2 import Boltz2

    orig_ps = Boltz2.predict_step
    state = {"n": 0, "done": False}

    def install(model):
        root = getattr(model, top) if top else model
        for name, m in root.named_modules():
            if type(m).__name__ != cls_name:
                continue
            fwd = m.forward

            def wrapped(*a, _m=m, _f=fwd, _name=name, **k):
                o = _f(*a, **k)
                state["n"] += 1
                if state["n"] == nth:
                    torch.save({"class": cls_name, "module_path": _name,
                                "init": {k2: v for k2, v in vars(_m).items()
                                         if isinstance(v, (int, float, bool, str))},
                                "args": to_cpu(a), "kwargs": to_cpu(k),
                                "state_dict": {k2: v.cpu() for k2, v in _m.state_dict().items()},
                                "out": to_cpu(o)}, out)
                    print(f"captured {cls_name} call {nth} at {_name}", flush=True)
                    raise _Done
                return o

            m.forward = wrapped

    def ps(self, *a, **k):
        if not state["done"]:
            install(self)
            state["done"] = True
        return orig_ps(self, *a, **k)

    Boltz2.predict_step = ps
    try:
        bm.predict.main(args=[*argv, "--num_workers", "0"], standalone_mode=False)
    except _Done:
        pass


if __name__ == "__main__":
    main()
