"""Command-line interface for cert_rnn.

    cert-rnn version
    cert-rnn demo                       run a tiny self-contained certification
    cert-rnn info  CHECKPOINT           inspect a checkpoint's tensors/shapes
    cert-rnn verify MODULE.pt --input X.npy [--spec margin|threshold] ...

`verify` expects a *pickled nn.Module* (saved with `torch.save(model, path)`,
not a bare state_dict) so the architecture is recoverable; it finds the
recurrent layer + optional linear head and certifies the chosen spec. For
anything beyond the common classifier case, use the Python API
(`cert_rnn.RNNModel` / `LSTMAutoencoder` + `certify`); see docs/quickstart.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_version(args) -> int:
    import cert_rnn

    print(f"cert-rnn {cert_rnn.__version__}")
    return 0


def _cmd_demo(args) -> int:
    """Self-contained: build a tiny LSTM classifier and certify it."""
    import numpy as np
    import torch
    import torch.nn as nn

    from cert_rnn import MarginSpec, RNNModel

    torch.manual_seed(0)
    D, H, T, C = 4, 6, 8, 3
    cell = nn.LSTMCell(D, H).double().eval()
    fc = nn.Linear(H, C).double().eval()
    model = RNNModel.from_torch(cell, fc)

    x = np.random.default_rng(0).standard_normal((T, D))
    true_class = int(np.argmax(model.reach_output(x, 0.0, "single_frame", 0).c))
    print(f"tiny LSTM  D={D} H={H} T={T} C={C}  predicted class={true_class}")

    result = model.certify(x, MarginSpec(true_class), n_iters=args.n_iters)
    print(result)
    return 0


class CheckpointError(Exception):
    """A user-facing checkpoint-loading failure (printed without traceback)."""


def _load_checkpoint(path: Path):
    if not path.exists():
        raise CheckpointError(f"no such file: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix in (".pt", ".pth"):
            import torch

            return torch.load(path, map_location="cpu", weights_only=False)
        if suffix == ".npz":
            import numpy as np

            return dict(np.load(path, allow_pickle=True))
        if suffix == ".mat":
            import scipy.io

            return scipy.io.loadmat(str(path))
    except AttributeError as e:
        raise CheckpointError(
            f"could not unpickle {path.name}: {e}.\n"
            "A full model saved with torch.save(model, ...) needs its class to be "
            "importable in this process. Prefer saving a state_dict, or build the "
            "model in Python and use the cert_rnn API (see docs/quickstart.md)."
        )
    raise CheckpointError(f"unsupported checkpoint type {suffix!r} (.pt/.pth/.npz/.mat)")


def _cmd_info(args) -> int:
    try:
        obj = _load_checkpoint(Path(args.checkpoint))
    except CheckpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    path = Path(args.checkpoint)

    import torch.nn as nn

    if isinstance(obj, nn.Module):
        print(f"{path.name}: nn.Module ({type(obj).__name__})")
        for name, p in obj.named_parameters():
            print(f"  {name:<40} {tuple(p.shape)}")
        return 0
    if hasattr(obj, "items"):
        print(f"{path.name}: dict with {len(obj)} entries")
        for k, v in obj.items():
            shape = getattr(v, "shape", None)
            print(f"  {str(k):<40} {tuple(shape) if shape is not None else type(v).__name__}")
        return 0
    print(f"{path.name}: {type(obj).__name__}")
    return 0


def _extract_classifier(module):
    """Find the first recurrent layer and first linear head in a module."""
    import torch.nn as nn

    rec = None
    head = None
    for m in module.modules():
        if rec is None and isinstance(m, (nn.LSTM, nn.LSTMCell, nn.RNN)):
            rec = m
        elif head is None and isinstance(m, nn.Linear):
            head = m
    return rec, head


def _cmd_verify(args) -> int:
    import numpy as np

    from cert_rnn import MarginSpec, RNNModel, ThresholdSpec

    try:
        obj = _load_checkpoint(Path(args.module))
    except CheckpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    import torch.nn as nn

    if not isinstance(obj, nn.Module):
        print(
            "error: verify needs a pickled nn.Module (torch.save(model, path)), "
            "not a state_dict. Build the model in Python and use cert_rnn.RNNModel "
            "/ LSTMAutoencoder + certify; see docs/quickstart.md.",
            file=sys.stderr,
        )
        return 2

    rec, head = _extract_classifier(obj)
    if rec is None:
        print("error: no nn.LSTM/nn.LSTMCell/nn.RNN found in the module", file=sys.stderr)
        return 2
    model = RNNModel.from_torch(rec.double(), head.double() if head is not None else None)

    x = np.load(args.input).astype(np.float64)
    if x.ndim != 2:
        print(f"error: --input must be a 2-D (T, D) array, got shape {x.shape}", file=sys.stderr)
        return 2
    print(f"model: D={model.D} H={model.H} L={model.L} head={model.has_head}; input {x.shape}")

    if args.spec == "margin":
        if not model.has_head:
            print("error: margin spec needs a linear head in the module", file=sys.stderr)
            return 2
        tc = args.true_class
        if tc is None:
            tc = int(np.argmax(model.reach_output(x, 0.0, args.threat_model, 0).c))
            print(f"  (true_class not given; using eps=0 argmax = {tc})")
        spec = MarginSpec(tc)
    else:  # threshold
        if args.upper is None and args.lower is None:
            print("error: threshold spec needs --upper and/or --lower", file=sys.stderr)
            return 2
        spec = ThresholdSpec(upper=args.upper, lower=args.lower)

    result = model.certify(
        x, spec,
        threat_model=args.threat_model,
        eps_init=args.eps_init,
        n_iters=args.n_iters,
    )
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cert-rnn", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)

    d = sub.add_parser("demo", help="run a tiny self-contained certification")
    d.add_argument("--n-iters", type=int, default=12)
    d.set_defaults(func=_cmd_demo)

    i = sub.add_parser("info", help="inspect a checkpoint's tensors/shapes")
    i.add_argument("checkpoint")
    i.set_defaults(func=_cmd_info)

    v = sub.add_parser("verify", help="certify a pickled nn.Module classifier")
    v.add_argument("module", help="path to a pickled nn.Module (.pt/.pth)")
    v.add_argument("--input", required=True, help="(T, D) input sequence, .npy")
    v.add_argument("--spec", choices=["margin", "threshold"], default="margin")
    v.add_argument("--true-class", type=int, default=None)
    v.add_argument("--upper", type=float, default=None)
    v.add_argument("--lower", type=float, default=None)
    v.add_argument("--threat-model", choices=["single_frame", "multi_frame"], default="single_frame")
    v.add_argument("--eps-init", type=float, default=0.5)
    v.add_argument("--n-iters", type=int, default=12)
    v.set_defaults(func=_cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
