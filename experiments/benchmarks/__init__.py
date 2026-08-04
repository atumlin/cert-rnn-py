"""Benchmark registry.

  ieee9            the four shipped IEEE-9 checkpoints (S/M/L/D)
  synth            deterministic synthetic scaling family (SynthSpec grid)
  smd / morris / skab   public TSAD datasets (data must be downloaded;
                        loaders print instructions when missing)
"""

from experiments.benchmarks import ieee9, public_tsad, synthetic  # noqa: F401
from experiments.benchmarks.synthetic import SynthSpec  # noqa: F401
