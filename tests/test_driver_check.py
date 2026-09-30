"""End-to-end checks of the Zone-A driver (examples/zoneA_lstm_ae/
verify_zoneA_v2.py) on a tiny per-step-code model: batch windows, attack
selection, fixed-eps verdicts, saved counterexamples, minimization, the
headline summary, and concentrated-attack radius stages."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cert_rnn import LSTMAutoencoder
from tests.soundness.test_sequence_decoder import _seq_ae

DRIVER = Path(__file__).resolve().parents[1] / "examples/zoneA_lstm_ae/verify_zoneA_v2.py"
SEED = 20260930


def _job(tmp_path, K=3, batch=True):
    _, ae = _seq_ae(5)
    rng = np.random.default_rng(SEED)
    anchors = rng.uniform(0.3, 0.7, (K, 5, 3))
    s = ae.score(anchors)
    tau = float(1.02 * s.max())             # all windows just below tau
    job = {"encoder": ae.encoder, "decoder": ae.decoder, "head": ae.head, "tau": tau}
    if batch:
        job["anchors"] = anchors
        job["anchor_info"] = [{"name": f"w{k}"} for k in range(K)]
    else:
        job["anchor"] = anchors[0]
    p = tmp_path / "job.pkl"
    pickle.dump(job, open(p, "wb"))
    return p, ae, anchors, tau


def _run(*args):
    r = subprocess.run([sys.executable, str(DRIVER), *map(str, args)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_batch_check_all_attacks_and_counterexamples(tmp_path):
    job, ae, anchors, tau = _job(tmp_path)
    out = tmp_path / "res.json"
    log = _run("--job", job, "--out", out, "--workers", 2, "--eps", "0.0005,0.3",
               "--attack", "whole,seconds,concentrated", "--seconds", "1,2",
               "--minimize-cex")
    res = json.load(open(out))
    assert "HEADLINE" in log and "SOUNDNESS FINDING" not in log
    for eps in ("0.0005", "0.3"):
        sm = res["summary"][f"check_eps_{eps}"]
        for fam in ("whole", "seconds", "concentrated"):
            assert sum(sm[fam].values()) == sm["windows"] - sm["already_alarm"]
            assert sm[fam]["UNSOUND"] == 0
    n_cex = 0
    for k in range(len(anchors)):
        st = res["anchors"][str(k)]["stages"]["check_eps_0.3"]
        assert len(st["one_second"]) == 5 and st["concentrated"]["seconds"] == [1, 2]
        entries = [("whole", st["whole_window"])] + \
                  [(f"sec{e['t']}", e) for e in st["one_second"]] + \
                  [("secs1-2", st["concentrated"])]
        for label, e in entries:
            for mode in ("box", "zono"):
                v = e[mode]["verdict_joint"]
                assert v in ("VERIFIED", "COUNTEREXAMPLE", "UNKNOWN")
                if e["attack_score"] > tau:
                    assert v == "COUNTEREXAMPLE"
            if "counterexample" not in e:
                assert e["attack_score"] <= tau
                continue
            n_cex += 1
            c = e["counterexample"]
            x = np.load(c["file"])
            d = np.abs(x - anchors[k])
            assert d.max() <= 0.3 * (1 + 1e-9)
            assert float(ae.score(x[None])[0]) > tau          # re-scores above tau
            if label.startswith("sec") and label != "secs1-2":
                t = int(label[3:])
                assert np.all(d[np.arange(5) != t] == 0)       # only that second moved
            if label == "secs1-2":
                assert np.all(d[[0, 3, 4]] == 0)
            xm = np.load(c["minimized"]["file"])
            dm = np.abs(xm - anchors[k])
            assert float(ae.score(xm[None])[0]) > tau
            assert np.all((dm > 0) <= (d > 0))                  # subset of the changes
            assert c["minimized"]["n_changed"] <= c["n_changed"]
    assert n_cex > 0, "eps=0.3 should produce counterexamples (test is vacuous)"


def test_single_window_check_keeps_legacy_layout(tmp_path):
    job, _, _, _ = _job(tmp_path, batch=False)
    out = tmp_path / "res.json"
    _run("--job", job, "--out", out, "--workers", 2, "--eps", "0.001",
         "--attack", "whole")
    st = json.load(open(out))["stages"]["check_eps_0.001"]
    assert "whole_window" in st and "one_second" not in st


def test_already_alarming_window_is_reported_not_checked(tmp_path):
    _, ae = _seq_ae(5)
    anchors = np.random.default_rng(SEED).uniform(0.3, 0.7, (2, 5, 3))
    s = ae.score(anchors)
    tau = float(s.mean())                   # one window above, one below
    p = tmp_path / "job.pkl"
    pickle.dump({"encoder": ae.encoder, "decoder": ae.decoder, "head": ae.head,
                 "tau": tau, "anchors": anchors}, open(p, "wb"))
    out = tmp_path / "res.json"
    log = _run("--job", p, "--out", out, "--workers", 2, "--eps", "0.001",
               "--attack", "whole")
    sm = json.load(open(out))["summary"]["check_eps_0.001"]
    assert sm["already_alarm"] == 1 and sum(sm["whole"].values()) == 1
    assert "already above tau" in log


def test_batch_radius_concentrated(tmp_path):
    job, _, anchors, _ = _job(tmp_path, K=2)
    out = tmp_path / "res.json"
    _run("--job", job, "--out", out, "--workers", 2, "--attack", "concentrated",
         "--seconds", "1,3")
    res = json.load(open(out))
    for k in range(2):
        st = res["anchors"][str(k)]["stages"]
        for n in ("box_seconds_1-3", "zono_seconds_1-3", "zono_joint_seconds_1-3"):
            assert st[n]["attack_seconds"] == [1, 3] and st[n]["radius"] > 0
        assert st["zono_joint_seconds_1-3"]["radius"] >= st["zono_seconds_1-3"]["radius"]
    assert res["summary_radius"]["zono_joint_seconds_1-3"]["windows"] == 2


def test_bad_attack_arguments(tmp_path):
    job, _, _, _ = _job(tmp_path)
    for extra in (["--attack", "concentrated"], ["--attack", "bogus"],
                  ["--attack", "concentrated", "--seconds", "9"]):
        r = subprocess.run([sys.executable, str(DRIVER), "--job", str(job), "--out",
                            str(tmp_path / "x.json"), "--eps", "0.01", *extra],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode != 0
