# experiments/ — comparison + scalability suites (ICLR 2027)

See [DESIGN.md](DESIGN.md) for the full experiment design, tool
landscape, and claim-to-suite mapping. This file is the operator manual.

## Layout

```
common.py          Record/JSONL results schema, env capture, timeouts
verifiers/         tool adapters behind one bisection harness
  certrnn-zono         this repo (the tool under study) [+ -masking]
  interval-ibp         from-scratch IBP baseline        [+ -masking]
  lirpa-ibp / lirpa-backward / lirpa-crown-optimized    auto_LiRPA
  pgd / pgd-masking    falsification (empirical upper bound)
benchmarks/        ieee9 (shipped), synthetic (generator), public_tsad (SMD/...)
suites/            e1_verifiability  e2_scalability  e3_detection  e4_tightness
tabulate.py        JSONL -> markdown tables
plots.py           JSONL -> cactus / scaling / radius figures (PDF)
run_all.sh         full battery, cheap -> expensive
```

## Quick start

```bash
# 2-minute sanity pass (S checkpoint, 1 frame, fast tools)
python -m experiments.suites.e1_verifiability \
    --sizes S --frames 0 --n-iters 8 --tools certrnn-zono interval-ibp pgd --tag smoke
python -m experiments.tabulate experiments/results/e1_smoke.jsonl

# full battery (hours; lirpa-backward dominates the cost)
experiments/run_all.sh
```

All timing runs pin BLAS to one thread (suites do this themselves).
Results append to `experiments/results/<suite>[_<tag>].jsonl`; each file
starts with an environment stamp (git rev, package versions, CPU).

## Public datasets

`benchmarks/data/` is git-ignored. Loaders raise instructions when data
is missing:

- SMD: clone https://github.com/NetManAIOps/OmniAnomaly, symlink its
  `ServerMachineDataset/` to `experiments/benchmarks/data/smd/`
- Morris power-attack PMU, SKAB: see `benchmarks/public_tsad.py`
  docstring (loaders not wired yet; SMD is the reference)

## Reading results

- `tabulate.py <files>` — benchmark×tool radius table (`^` marks attack
  upper bounds)
- `tabulate.py --certified-at 0.01` — certified-robust rate at eps
- `tabulate.py --scaling` — sec/bound-call rows (E2)
- `plots.py {cactus,scaling,radius} <files>` — paper figures
