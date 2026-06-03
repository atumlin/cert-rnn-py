"""Build a comparison table of LSTM-AE Spec C results (Python vs MATLAB)
for the four shipped sizes (S, M, L, D).

Requires that examples/lstm_ae_ieee9/verify_all.py has been run for
the sizes you want to tabulate (results go to results/*.npz).

Run:
    python scripts/tabulate_lstm_ae.py
    python scripts/tabulate_lstm_ae.py --md   # markdown output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "lstm_ae_ieee9"
PY_DIR = EXAMPLES / "results"
MAT_DIR = EXAMPLES / "matlab_results"

SIZES = ["S", "M", "L", "D"]


def _f(x):
    return float(np.asarray(x).item())


def load_py(size: str) -> dict | None:
    p = PY_DIR / f"certrnn_lstm_ae_{size}.npz"
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=False))
    return d


def load_mat(size: str) -> dict | None:
    p = MAT_DIR / f"certrnn_lstm_ae_{size}.mat"
    if not p.exists():
        return None
    return scipy.io.loadmat(str(p))


def per_size_summary(size: str) -> dict:
    py = load_py(size)
    mat = load_mat(size)
    if py is None and mat is None:
        return {"size": size, "status": "no data"}

    out = {"size": size, "status": "ok"}

    # MATLAB stats
    if mat is not None:
        out["H_mat"] = int(mat["H"].item())
        out["T_mat"] = int(mat["T"].item())
        out["D_mat"] = int(mat["D"].item())
        out["cert_radius_mat"] = _f(mat["cert_radius"])
        out["eps_mean_mat"] = float(np.mean(mat["eps_per_frame"].flatten()))
        out["eps_per_frame_mat"] = mat["eps_per_frame"].flatten().astype(np.float64)
        out["total_time_mat_s"] = _f(mat["total_time_s"])
        out["mean_time_mat_s"] = float(np.mean(mat["times_per_frame"].flatten()))

    # Python stats
    if py is not None:
        out["cert_radius_py"] = _f(py["cert_radius"])
        out["eps_per_frame_py"] = py["eps_per_frame"]
        out["eps_mean_py"] = float(np.mean(py["eps_per_frame"]))
        out["total_time_py_s"] = _f(py["total_time_s"])
        out["mean_time_py_s"] = float(np.mean(py["times_per_frame"]))
        out["H_py"] = int(py["H"].item()) if hasattr(py.get("H", 0), "item") else int(py["H"])

    # Differences when both present
    if py is not None and mat is not None:
        diff = out["eps_per_frame_py"] - out["eps_per_frame_mat"]
        out["max_abs_eps_diff"] = float(np.max(np.abs(diff)))
        out["max_signed_eps_diff"] = float(diff[np.argmax(np.abs(diff))])
        out["frames_py_geq_mat"] = int(np.sum(diff >= -1e-10))
        out["speed_ratio"] = out["mean_time_py_s"] / out["mean_time_mat_s"]

    return out


def format_table_plain(rows: list[dict]) -> str:
    lines = []
    h = (
        f"{'size':<5} {'H':>3} {'cert_eps':>16} {'mean_eps':>16} {'max|Δε|':>10} "
        f"{'frames py≥mat':>14} {'time/frame (s)':>22} {'speedup':>9}"
    )
    lines.append(h)
    lines.append(
        f"{'':5} {'':3} {'py    |  matlab':>16} {'py    |  matlab':>16} "
        f"{'':10} {'':14} {'py    |  matlab':>22} {'py/mat':>9}"
    )
    lines.append("-" * len(h))
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['size']:<5} {r['status']}")
            continue
        cert_py = f"{r.get('cert_radius_py', float('nan')):.5f}"
        cert_mat = f"{r.get('cert_radius_mat', float('nan')):.5f}"
        mean_py = f"{r.get('eps_mean_py', float('nan')):.5f}"
        mean_mat = f"{r.get('eps_mean_mat', float('nan')):.5f}"
        max_diff = f"{r.get('max_abs_eps_diff', float('nan')):.2e}"
        frames = (
            f"{r.get('frames_py_geq_mat', '?')}/{len(r.get('eps_per_frame_py', []))}"
            if r.get('eps_per_frame_py') is not None else "?"
        )
        tp = f"{r.get('mean_time_py_s', float('nan')):.2f}"
        tm = f"{r.get('mean_time_mat_s', float('nan')):.2f}"
        sp = f"{r.get('speed_ratio', float('nan')):.1f}x"
        H = r.get("H_mat") or r.get("H_py", "?")
        lines.append(
            f"{r['size']:<5} {H:>3} {cert_py:>7} | {cert_mat:<7} "
            f"{mean_py:>7} | {mean_mat:<7} {max_diff:>10} "
            f"{frames:>14} {tp:>7} | {tm:<7} {sp:>9}"
        )
    return "\n".join(lines)


def format_table_markdown(rows: list[dict]) -> str:
    lines = [
        "| size | H | cert_eps py | cert_eps matlab | mean ε py | mean ε matlab | max\\|Δε\\| | frames py≥mat | s/frame py | s/frame matlab | py/mat |",
        "|------|---|-------------|-----------------|-----------|---------------|----------|---------------|------------|----------------|--------|",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['size']} | | _{r['status']}_ |")
            continue
        H = r.get("H_mat") or r.get("H_py", "?")
        lines.append(
            f"| {r['size']} | {H} "
            f"| {r.get('cert_radius_py', float('nan')):.5f} "
            f"| {r.get('cert_radius_mat', float('nan')):.5f} "
            f"| {r.get('eps_mean_py', float('nan')):.5f} "
            f"| {r.get('eps_mean_mat', float('nan')):.5f} "
            f"| {r.get('max_abs_eps_diff', float('nan')):.2e} "
            f"| {r.get('frames_py_geq_mat', '?')}/{len(r.get('eps_per_frame_py', []))} "
            f"| {r.get('mean_time_py_s', float('nan')):.2f} "
            f"| {r.get('mean_time_mat_s', float('nan')):.2f} "
            f"| {r.get('speed_ratio', float('nan')):.1f}× |"
        )
    return "\n".join(lines)


def format_per_frame_md(size: str, r: dict) -> str:
    if r["status"] != "ok" or "eps_per_frame_py" not in r or "eps_per_frame_mat" not in r:
        return f"_{size}: no per-frame comparison available_"
    py_arr = r["eps_per_frame_py"]
    mat_arr = r["eps_per_frame_mat"]
    if py_arr is None or mat_arr is None:
        return f"_{size}: no per-frame comparison available_"
    lines = [
        f"\n### {size} (H={r.get('H_mat') or r.get('H_py')}) per-frame",
        "",
        "| t | py | matlab | py − matlab |",
        "|---|----|--------|-------------|",
    ]
    for t, (py_v, mat_v) in enumerate(zip(py_arr, mat_arr)):
        lines.append(
            f"| {t} | {float(py_v):.5f} | {float(mat_v):.5f} | {float(py_v - mat_v):+.3e} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--md", action="store_true")
    p.add_argument("--per-frame", action="store_true",
                   help="include per-frame eps table per size (long)")
    p.add_argument("--out", default=None, help="write markdown to this file")
    args = p.parse_args(argv)

    rows = [per_size_summary(s) for s in SIZES]

    if args.md or args.out:
        body = format_table_markdown(rows)
        if args.per_frame:
            body += "\n"
            for size, r in zip(SIZES, rows):
                body += "\n" + format_per_frame_md(size, r)
        if args.out:
            Path(args.out).write_text(body + "\n")
            print(f"wrote {args.out}")
        else:
            print(body)
    else:
        print(format_table_plain(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
