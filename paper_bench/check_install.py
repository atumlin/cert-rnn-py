"""Installation check: imports, GPU, and a tiny end-to-end query per tool.

    python -m paper_bench.check_install            # all checks
    python -m paper_bench.check_install --no-tools # imports + GPU only
"""

from __future__ import annotations

import argparse
import importlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tools", action="store_true")
    args = ap.parse_args()
    ok = True
    for mod in ("numpy", "torch", "auto_LiRPA", "abcrown", "onnx", "onnxruntime",
                "cert_rnn", "pandapower", "matplotlib", "yaml", "sklearn"):
        try:
            m = importlib.import_module(mod)
            print(f"  [ok]   {mod:12s} {getattr(m, '__version__', '')}")
        except Exception as e:                      # noqa: BLE001
            ok = False
            print(f"  [FAIL] {mod:12s} {type(e).__name__}: {e}")
    import torch
    print(f"  [info] python {sys.version.split()[0]} | CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if ok and not args.no_tools:
        from paper_bench.tools import smoke_test
        ok = smoke_test() and ok
    print("INSTALL OK" if ok else "INSTALL INCOMPLETE -- see [FAIL] lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
