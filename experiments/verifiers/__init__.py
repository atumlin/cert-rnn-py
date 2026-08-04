"""Verifier registry. `get_verifier(name)` -> adapter instance.

Lower-bound (sound) tools:
    certrnn-zono        this repo's Cert-RNN zonotope engine
    interval-ibp        interval bound propagation baseline
    lirpa-ibp           auto_LiRPA interval mode on the unrolled graph
    lirpa-backward      auto_LiRPA CROWN (backward linear relaxation)
    lirpa-crown-optimized  alpha-CROWN (optimized relaxation)
Upper-bound (attack) tools:
    pgd                 multi-restart PGD falsification
Masking-property variants (score >= tau on anomalous anchors) append
"-masking"."""

from __future__ import annotations

from experiments.verifiers.base import AdapterResult, Case  # noqa: F401


def get_verifier(name: str):
    if name == "certrnn-zono":
        from experiments.verifiers.certrnn_zono import CertRnnZono
        return CertRnnZono()
    if name == "certrnn-zono-masking":
        from experiments.verifiers.certrnn_zono import CertRnnZonoMasking
        return CertRnnZonoMasking()
    if name == "interval-ibp":
        from experiments.verifiers.interval import IntervalIBP
        return IntervalIBP()
    if name == "interval-ibp-masking":
        from experiments.verifiers.interval import IntervalIBPMasking
        return IntervalIBPMasking()
    if name.startswith("lirpa-"):
        from experiments.verifiers.lirpa import LirpaBound
        rest = name[len("lirpa-"):]
        masking = rest.endswith("-masking")
        if masking:
            rest = rest[: -len("-masking")]
        method = {"ibp": "IBP", "backward": "backward",
                  "crown-optimized": "CROWN-Optimized"}[rest]
        return LirpaBound(method=method, masking=masking)
    if name == "pgd":
        from experiments.verifiers.pgd import PGDAttack
        return PGDAttack(direction="up")
    if name == "pgd-masking":
        from experiments.verifiers.pgd import PGDAttack
        return PGDAttack(direction="down")
    raise KeyError(f"unknown verifier {name!r}")


LOWER_BOUND_TOOLS = ["certrnn-zono", "interval-ibp", "lirpa-ibp",
                     "lirpa-backward"]
ATTACK_TOOLS = ["pgd"]
