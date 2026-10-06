"""Step 1: benchmark data. Generates the grid stand-ins (skips finished ones);
for Zone-A only checks the given folder.

    python -m paper_bench.make_data                      # all grid benchmarks
    python -m paper_bench.make_data --bench zoneA --data-dir /path/to/zoneA
"""

from __future__ import annotations

import argparse
import os
import shutil

from paper_bench.common import data_dir, load_config
from paper_bench.data import LABELS_F, TEST_F, TRAIN_F, generate_grid, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", help="benchmarks (default: all grid benchmarks)")
    ap.add_argument("--data-dir", help="data folder for source: dir benchmarks (Zone-A)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    benches = args.bench or [b for b, s in cfg["benchmarks"].items() if s["source"] == "grid"]
    done_cases = set()
    for b in benches:
        spec = cfg["benchmarks"][b]
        if spec["source"] == "dir":
            d = data_dir(cfg, b, args.data_dir)
            missing = [f for f in (TRAIN_F, TEST_F, LABELS_F) if not (d / f).exists()]
            if missing:
                raise SystemExit(f"{b}: missing {missing} in {d}")
            p = prepare(d, spec["window"])
            print(f"[{b}] {d}: {len(p.features)} features | windows: train {len(p.train)}, val {len(p.val)}, "
                  f"test normal {len(p.test_normal)}, test anomaly {len(p.test_anomaly)}")
            continue
        final = data_dir(cfg, b)
        if spec["case"] in done_cases or (final / LABELS_F).exists():
            print(f"[{b}] data ready: {final}")
            done_cases.add(spec["case"])
            continue
        tmp = final.with_name(final.name + ".partial")          # renamed only when complete
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[{b}] generating {spec['case']} -> {final}", flush=True)
        g = cfg["grid"]
        generate_grid(spec["case"], tmp, g["baseline_s"], g["scenario_s"], g["n_events"], g["seed"])
        os.replace(tmp, final)
        done_cases.add(spec["case"])
        print(f"[{b}] done", flush=True)


if __name__ == "__main__":
    main()
