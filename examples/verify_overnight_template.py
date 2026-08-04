"""Overnight LSTM-AE certification driver -- copy next to your notebook.

Loads the model from a plain state_dict checkpoint (no model class
needed), rebuilds the encoder/decoder stacks, gates on preflight +
smoke test, then runs a RESUMABLE per-frame certification loop that
checkpoints results after every frame and heartbeats to stdout every
reach (~20s at H=55, T=64).

One-time export cell in your notebook (all the script needs):

    torch.save(model.state_dict(), "lstm_ae_checkpoint.pt")
    np.savez("verify_inputs.npz", anchor=anchor, tau=float(tau))

Adjust ENC_LAYERS / DEC_LAYERS / HEAD below if your submodule names
differ from encoder.rnn1 / encoder.rnn2 / decoder.rnn1 / decoder.rnn2 /
decoder.output_layer.

Launch detached so a VSCode/SSH disconnect cannot kill it:

    tmux new -s verify
    python -u verify_overnight_template.py 2>&1 | tee verify.log
    # detach: Ctrl-b then d          reattach: tmux attach -t verify

Monitor from any terminal:

    tail -f verify.log         # a new line every ~20s while healthy
    ls -l verify.log           # mtime older than ~2 min => hung

Crash/reboot recovery: just run it again -- frames already in the
results file are skipped.

Options:
    --threat-model multi_frame     one joint bisection (~minutes, smaller
                                   radius) instead of per-frame (~hours)
    --frames 0 16 32 48            single_frame subset instead of all T
    --n-iters 12                   bisection depth (fewer = faster, coarser)
    --eps-init 0.5                 bisection starting radius; use a smaller
                                   value (e.g. 0.1) when your data scale
                                   makes 0.5 an enormous perturbation
    --smoke-only                   preflight + smoke test, then exit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cert_rnn import LSTMAutoencoder, ReconErrorSpec

# ---- paths + your model's state_dict layout --------------------------------
CKPT = "lstm_ae_checkpoint.pt"
INPUTS = "verify_inputs.npz"
RESULTS = "results_overnight.npz"

ENC_LAYERS = ["encoder.rnn1", "encoder.rnn2"]   # bottom -> top
DEC_LAYERS = ["decoder.rnn1", "decoder.rnn2"]
HEAD = "decoder.output_layer"


# ---- rebuild torch cells straight from the state_dict ----------------------
def cell_from_sd(sd: dict, prefix: str) -> nn.LSTMCell:
    """One single-layer nn.LSTM's weights (weight_ih_l0, ...) -> nn.LSTMCell."""
    W_ih, W_hh = sd[f"{prefix}.weight_ih_l0"], sd[f"{prefix}.weight_hh_l0"]
    c = nn.LSTMCell(W_ih.shape[1], W_hh.shape[1]).to(W_ih.dtype)
    with torch.no_grad():
        c.weight_ih.copy_(W_ih)
        c.weight_hh.copy_(W_hh)
        c.bias_ih.copy_(sd[f"{prefix}.bias_ih_l0"])
        c.bias_hh.copy_(sd[f"{prefix}.bias_hh_l0"])
    return c


def head_from_sd(sd: dict, prefix: str) -> nn.Linear:
    W = sd[f"{prefix}.weight"]
    h = nn.Linear(W.shape[1], W.shape[0]).to(W.dtype)
    with torch.no_grad():
        h.weight.copy_(W)
        h.bias.copy_(sd[f"{prefix}.bias"])
    return h


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threat-model", default="single_frame",
                   choices=["single_frame", "multi_frame"])
    p.add_argument("--frames", nargs="+", type=int, default=None,
                   help="single_frame: subset of frame indices (default: all)")
    p.add_argument("--n-iters", type=int, default=12)
    p.add_argument("--eps-init", type=float, default=0.5)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()

    log = lambda s: print(s, flush=True)  # noqa: E731  (flush => live log)

    # ---- load ----------------------------------------------------------------
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    inputs = np.load(INPUTS)
    anchor = np.asarray(inputs["anchor"], dtype=np.float64)
    tau = float(inputs["tau"])

    ae = LSTMAutoencoder.from_torch(
        [cell_from_sd(sd, name) for name in ENC_LAYERS],
        [cell_from_sd(sd, name) for name in DEC_LAYERS],
        head_from_sd(sd, HEAD),
    )
    log(f"loaded: D={ae.D} H={ae.H} L_enc={ae.encoder['L']} "
        f"L_dec={ae.decoder['L']}  anchor {anchor.shape}  tau={tau:.6g}")

    # ---- gate: preflight + smoke test (fail fast, before the long part) ------
    # (torch-model score parity should be checked once in the notebook, where
    # the original model object lives; here preflight covers the anchor side
    # and smoke_test re-checks engine-vs-concrete parity.)
    report = ae.preflight(anchor, tau=tau, title="overnight run")
    log(str(report))
    if not report.ok:
        log("ABORT: preflight failed")
        return 1
    smoke = ae.smoke_test(anchor, tau, n_frames=16)
    log(f"smoke: {smoke}")
    if not smoke["ok"]:
        log("ABORT: smoke test failed")
        return 1
    log(f"forecast: multi_frame ~{smoke['est_certify_multi_frame_s'] / 60:.1f} min, "
        f"single_frame (all frames) ~{smoke['est_certify_single_frame_s'] / 3600:.1f} h")
    if args.smoke_only:
        return 0

    # ---- multi_frame: one joint bisection, minutes ---------------------------
    if args.threat_model == "multi_frame":
        res = ae.certify(anchor, ReconErrorSpec(tau), threat_model="multi_frame",
                         n_iters=args.n_iters, eps_init=args.eps_init, progress=log)
        log(str(res))
        np.savez(RESULTS, radius=res.radius, tau=tau, threat_model="multi_frame",
                 seconds=res.seconds)
        log(f"saved {RESULTS}")
        return 0

    # ---- single_frame: resumable per-frame loop, checkpoint every frame ------
    T = anchor.shape[0]
    frames = list(range(T)) if args.frames is None else args.frames
    radii = np.full(T, np.nan)
    seconds = np.full(T, np.nan)
    out = Path(RESULTS)
    if out.exists():
        d = np.load(out)
        if "radii" in d:
            radii, seconds = d["radii"], d["seconds"]
            log(f"resuming: {int(np.sum(np.isfinite(radii)))} frame(s) already done")

    for t in frames:
        if np.isfinite(radii[t]):
            continue
        res = ae.certify(anchor, ReconErrorSpec(tau), frames=[t],
                         n_iters=args.n_iters, eps_init=args.eps_init, progress=log)
        radii[t], seconds[t] = res.per_frame[t], res.seconds
        np.savez(out, radii=radii, seconds=seconds, tau=tau,
                 threat_model="single_frame")
        n_done = int(np.sum(np.isfinite(radii)))
        log(f"=== checkpointed {n_done}/{len(frames)} frame(s) -> {out} ===")

    done = radii[frames]
    log(f"DONE  certified radius (min over {len(frames)} frame(s)): "
        f"{np.nanmin(done):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
