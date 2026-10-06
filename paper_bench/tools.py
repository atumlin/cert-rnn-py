"""Tool adapters. Every tool answers the same query -- "for the window x and
every input whose attacked readings move by at most eps, is score <= tau?" --
and returns a verdict dict:

    status   verified | unknown | falsified | timeout | error
    ub       the tool's upper bound on the score (verifiers), or the best
             score found (PGD)
    wall/cpu seconds, plus tool-specific info

Verifiers: Cert-RNN, Cert-RNN+ (zonotopes, this repo), IBP / CROWN /
alpha-CROWN (auto_LiRPA), alpha-beta-CROWN + GenBaB (subprocess). PGD is an
attack: it can only falsify (find a real input with score > tau).

All tools verify the same trained weights: cert-rnn dicts for the zonotope
engine, and the explicit-op `ScoreGraph` built from those same dicts for the
torch-based tools.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from paper_bench.common import HERE, Stopwatch, Timeout, time_limit
from paper_bench.model import ScoreGraph, V3AE, to_certrnn


# --------------------------------------------------------------------------- #
# model context: everything a tool needs about one trained model
# --------------------------------------------------------------------------- #
class ModelContext:
    def __init__(self, mdir: Path):
        import json
        self.dir = Path(mdir)
        self.meta = json.load(open(self.dir / "meta.json"))
        self.T, self.D, self.tau = self.meta["T"], self.meta["D"], float(self.meta["tau"])
        a = np.load(self.dir / "anchors.npz")
        self.windows = a["windows"]                       # (n, T, D) float64
        m = V3AE(self.D, self.meta["layers"])
        m.load_state_dict(torch.load(self.dir / "model.pt", weights_only=False))
        self.ae = to_certrnn(m)                           # cert-rnn dicts (fp64)
        self._graphs = {}
        # every tool must see the same function: explicit-op graph == cert-rnn model
        with torch.no_grad():
            g = self.graph("cpu", torch.float64)(torch.as_tensor(self.windows.reshape(len(self.windows), -1)))
        diff = float(np.max(np.abs(g.numpy().ravel() - self.ae.score(self.windows))))
        assert diff < 1e-9, f"score graph != cert-rnn model (max |diff| {diff:.2e})"

    def graph(self, device: str, dtype=torch.float32) -> ScoreGraph:
        k = (device, dtype)
        if k not in self._graphs:
            self._graphs[k] = ScoreGraph(self.ae, self.T, dtype).to(device).eval()
        return self._graphs[k]

    def mask(self, frames) -> np.ndarray:
        m = np.zeros((self.T, self.D))
        m[list(frames)] = 1.0
        return m


def _verdict(status, ub=None, sw=None, **info):
    out = {"status": status, "ub": None if ub is None else float(ub)}
    if sw is not None:
        out.update(wall=round(sw.wall, 4), cpu=round(sw.cpu, 4))
    out.update(info)
    return out


def _guard(fn, budget):
    """Run fn() under the per-query budget; map timeouts/exceptions to verdicts."""
    with Stopwatch() as sw:
        try:
            with time_limit(budget):
                status, ub, info = fn()
        except Timeout:
            status, ub, info = "timeout", None, {}
        except Exception as e:                            # noqa: BLE001 -- tool failure, not ours
            status, ub, info = "error", None, {"error": f"{type(e).__name__}: {e}",
                                               "trace": traceback.format_exc(limit=3)[-800:]}
    return _verdict(status, ub, sw, **info)


# --------------------------------------------------------------------------- #
# Cert-RNN / Cert-RNN+ (zonotopes)
# --------------------------------------------------------------------------- #
class CertRNN:
    def __init__(self, name, cfg):
        self.name, self.budget = name, cfg["budget"]
        self.gates, self.score = cfg["gates"], cfg["score"]
        self.device = "cpu"

    def check(self, ctx: ModelContext, w: int, frames, eps: float):
        from cert_rnn.transformers import bilinear_mode
        from cert_rnn.verify import lstm_ae_reach, spec_c_score_ub, spec_c_score_ub_joint
        x = ctx.windows[w]
        if len(frames) == ctx.T:
            tm, tp = "multi_frame", None
        elif len(frames) == 1:
            tm, tp = "single_frame", int(frames[0])
        else:
            tm, tp = "frame_set", tuple(int(f) for f in frames)

        def run():
            with bilinear_mode(self.gates):
                zxh, zx = lstm_ae_reach(ctx.ae.encoder, ctx.ae.decoder, ctx.ae.head, x, eps, tm, tp)
            ub = (spec_c_score_ub_joint if self.score == "joint" else spec_c_score_ub)(zxh, zx)
            return ("verified" if ub <= ctx.tau else "unknown"), ub, {}
        return _guard(run, self.budget)


# --------------------------------------------------------------------------- #
# auto_LiRPA: IBP / CROWN / alpha-CROWN on the score graph
# --------------------------------------------------------------------------- #
class Lirpa:
    def __init__(self, name, cfg, device):
        self.name, self.budget, self.method, self.device = name, cfg["budget"], cfg["method"], device
        self._bm = {}

    def _bounded(self, ctx):
        from auto_LiRPA import BoundedModule
        k = id(ctx)
        if k not in self._bm:
            self._bm.clear()
            g = ctx.graph(self.device)
            dummy = torch.zeros(1, ctx.T * ctx.D, device=self.device)
            self._bm[k] = BoundedModule(g, dummy, device=self.device, verbose=0)
        return self._bm[k]

    def check(self, ctx: ModelContext, w: int, frames, eps: float):
        from auto_LiRPA import BoundedTensor, PerturbationLpNorm

        def run():
            bm = self._bounded(ctx)
            x0 = torch.as_tensor(ctx.windows[w].reshape(1, -1), dtype=torch.float32, device=self.device)
            m = torch.as_tensor(ctx.mask(frames).reshape(1, -1), dtype=torch.float32, device=self.device)
            ptb = PerturbationLpNorm(norm=np.inf, x_L=x0 - eps * m, x_U=x0 + eps * m)
            _, ub = bm.compute_bounds(x=(BoundedTensor(x0, ptb),), method=self.method,
                                      bound_lower=False, bound_upper=True)
            ub = float(ub.max().item())
            return ("verified" if ub <= ctx.tau else "unknown"), ub, {}
        return _guard(run, self.budget)


# --------------------------------------------------------------------------- #
# alpha-beta-CROWN + GenBaB (one subprocess per query)
# --------------------------------------------------------------------------- #
class _FixedScore(torch.nn.Module):
    """Score graph whose input is ONLY the attacked readings (B, k*D); the other
    seconds enter as constants through a selector matmul (batch-safe)."""

    def __init__(self, g: ScoreGraph, anchor: np.ndarray, frames):
        super().__init__()
        self.g = g
        T, D = anchor.shape
        idx = [t * D + d for t in frames for d in range(D)]
        sel = np.zeros((len(idx), T * D), dtype=np.float32)
        sel[np.arange(len(idx)), idx] = 1.0
        base = anchor.astype(np.float32).ravel().copy()
        base[idx] = 0.0
        self.register_buffer("sel", torch.as_tensor(sel))
        self.register_buffer("base", torch.as_tensor(base).unsqueeze(0))

    def forward(self, z):
        return self.g(z @ self.sel + self.base)


class ABCrown:
    """Each query: export ONNX (score folded in) + VNN-LIB spec 'score >= tau'
    (the violation); alpha-beta-CROWN answers unsat (= verified), sat
    (= counterexample), timeout or unknown. A crash is an error verdict."""

    def __init__(self, name, cfg, device):
        self.name, self.budget, self.device = name, cfg["budget"], device
        self.encoding = cfg.get("encoding", "fixed")
        self.abc_dir = Path(os.environ.get("ABCROWN_DIR", HERE / "third_party" / "alpha-beta-CROWN"))
        self.config = HERE / "abcrown.yaml"

    def _files(self, ctx, w, frames, eps, tmp: Path):
        x = ctx.windows[w]
        g = ctx.graph("cpu")
        if self.encoding == "fixed":
            net = _FixedScore(g, x, frames).eval()
            center = x[list(frames)].ravel()
            lo, hi = center - eps, center + eps
        else:
            net = g
            m = ctx.mask(frames).ravel()
            lo, hi = x.ravel() - eps * m, x.ravel() + eps * m
        onnx_p, spec_p = tmp / "net.onnx", tmp / "spec.vnnlib"
        torch.onnx.export(net, torch.zeros(1, len(lo)), onnx_p, input_names=["x"], output_names=["score"],
                          dynamic_axes={"x": {0: "batch"}, "score": {0: "batch"}}, opset_version=17, dynamo=False)
        with open(spec_p, "w") as f:
            for i in range(len(lo)):
                f.write(f"(declare-const X_{i} Real)\n")
            f.write("(declare-const Y_0 Real)\n")
            for i, (a, b) in enumerate(zip(lo, hi)):
                f.write(f"(assert (>= X_{i} {a:.10g}))\n(assert (<= X_{i} {b:.10g}))\n")
            f.write(f"(assert (>= Y_0 {ctx.tau:.10g}))\n")      # the violation: an alarm
        return onnx_p, spec_p

    def check(self, ctx: ModelContext, w: int, frames, eps: float):
        with Stopwatch() as sw, tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            try:
                onnx_p, spec_p = self._files(ctx, w, frames, eps, tmp)
                res = tmp / "out.txt"
                cmd = [sys.executable, str(self.abc_dir / "complete_verifier" / "abcrown.py"),
                       "--config", str(self.config), "--onnx_path", str(onnx_p),
                       "--vnnlib_path", str(spec_p), "--results_file", str(res),
                       "--device", self.device, "--timeout", str(self.budget)]
                env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
                p = subprocess.run(cmd, cwd=self.abc_dir / "complete_verifier", env=env, capture_output=True,
                                   text=True, timeout=self.budget + 300)
                verdict = res.read_text().strip().split("\n")[0] if res.exists() else ""
                tail = (p.stdout + p.stderr)[-1500:]
            except subprocess.TimeoutExpired:
                verdict, tail = "timeout", "killed after budget + 300 s"
        info = {"encoding": self.encoding, "abcrown_verdict": verdict}
        if verdict.startswith("unsat"):
            status = "verified"
        elif verdict.startswith("sat"):
            status = "falsified"
        elif verdict.startswith("timeout"):
            status = "timeout"
        elif verdict.startswith("unknown"):
            status = "unknown"
        else:
            status = "error"
            info["error"] = tail
        return _verdict(status, None, sw, **info)


# --------------------------------------------------------------------------- #
# PGD (attack -- can only falsify)
# --------------------------------------------------------------------------- #
class PGD:
    def __init__(self, name, cfg, device):
        self.name, self.budget, self.device = name, cfg["budget"], device
        self.restarts, self.steps = cfg["restarts"], cfg["steps"]

    def attack(self, ctx: ModelContext, w: int, frames, eps: float):
        g = ctx.graph(self.device)
        x0 = torch.as_tensor(ctx.windows[w].reshape(1, -1), dtype=torch.float32, device=self.device)
        m = torch.as_tensor(ctx.mask(frames).reshape(1, -1), dtype=torch.float32, device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(0)
        a = x0.repeat(self.restarts, 1)
        x = a + eps * m * (2 * torch.rand(a.shape, generator=gen, device=self.device) - 1)
        x[0], x[1] = (x0 + eps * m)[0], (x0 - eps * m)[0]           # include two corners
        best, best_x = -np.inf, None
        for _ in range(self.steps):
            x.requires_grad_(True)
            s = g(x).squeeze(1)
            i = int(s.argmax())
            if float(s[i]) > best:
                best, best_x = float(s[i]), x[i].detach().clone()
            grad, = torch.autograd.grad(s.sum(), x)
            with torch.no_grad():
                x = a + torch.clamp(x + 0.25 * eps * m * grad.sign() - a, -eps, eps) * m
        with torch.no_grad():
            s = g(x).squeeze(1)
            if float(s.max()) > best:
                best, best_x = float(s.max()), x[int(s.argmax())].clone()
        return best, best_x.cpu().numpy().reshape(ctx.T, ctx.D)

    def check(self, ctx: ModelContext, w: int, frames, eps: float):
        holder = {}

        def run():
            best, xb = self.attack(ctx, w, frames, eps)
            holder["x"] = xb
            return ("falsified" if best > ctx.tau else "no_violation"), best, {}
        v = _guard(run, self.budget)
        v["_x"] = holder.get("x")                         # counterexample input (kept local, not logged)
        return v


# --------------------------------------------------------------------------- #
def make_tool(name: str, tcfg: dict, hardware: str = "recommended"):
    device = "cpu" if hardware == "fair" else tcfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    kind = tcfg["kind"]
    if kind == "certrnn":
        return CertRNN(name, tcfg)
    if kind == "lirpa":
        return Lirpa(name, tcfg, device)
    if kind == "abcrown":
        return ABCrown(name, tcfg, device)
    if kind == "pgd":
        return PGD(name, tcfg, device)
    raise ValueError(kind)


def smoke_test() -> bool:
    """Tiny model, one query per tool; returns True if every tool ran."""
    import json
    from paper_bench.model import code_size
    from paper_bench.common import load_config
    cfg = load_config()
    torch.manual_seed(0)
    T, D = 4, 6
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        m = V3AE(D, 1)
        torch.save(m.state_dict(), d / "model.pt")
        x = np.random.default_rng(0).uniform(0.3, 0.7, (1, T, D))
        s = float(((m(torch.as_tensor(x, dtype=torch.float32)).detach().numpy() - x) ** 2).mean())
        np.savez(d / "anchors.npz", windows=x)
        json.dump({"T": T, "D": D, "H": code_size(D), "layers": 1, "tau": 2 * s}, open(d / "meta.json", "w"))
        ctx = ModelContext(d)
        ok = True
        for name, tcfg in cfg["tools"].items():
            tcfg = dict(tcfg, budget=min(tcfg["budget"], 120))
            v = make_tool(name, tcfg).check(ctx, 0, [1], 0.001)
            good = v["status"] not in ("error", "timeout")
            ok &= good
            print(f"  [{'ok' if good else 'FAIL'}]   tool {name:13s} -> {v['status']}"
                  + ("" if good else f"  {v.get('error', '')[:300]}"))
    return ok
