"""CLI smoke tests: version, demo, info, verify, and graceful errors."""

import numpy as np
import torch
import torch.nn as nn

from cert_rnn.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "cert-rnn" in capsys.readouterr().out


def test_demo(capsys):
    assert main(["demo", "--n-iters", "4"]) == 0
    out = capsys.readouterr().out
    assert "CertResult" in out and "certified" in out


def test_info_state_dict(tmp_path, capsys):
    p = tmp_path / "sd.pt"
    torch.save(nn.LSTM(3, 4).double().state_dict(), p)
    assert main(["info", str(p)]) == 0
    assert "weight_ih_l0" in capsys.readouterr().out


def test_info_missing_file(capsys):
    assert main(["info", "/nonexistent/x.pt"]) == 2
    assert "error" in capsys.readouterr().err


def test_verify_threshold(tmp_path, capsys):
    mp = tmp_path / "m.pt"
    torch.save(nn.LSTM(3, 4).double(), mp)   # nn.LSTM class is importable
    xp = tmp_path / "x.npy"
    np.save(xp, np.zeros((5, 3)))
    rc = main([
        "verify", str(mp), "--input", str(xp),
        "--spec", "threshold", "--upper", "10", "--lower", "-10", "--n-iters", "4",
    ])
    assert rc == 0
    assert "CertResult" in capsys.readouterr().out


def test_verify_statedict_is_graceful_error(tmp_path, capsys):
    sp = tmp_path / "sd.pt"
    torch.save(nn.LSTM(3, 4).double().state_dict(), sp)
    xp = tmp_path / "x.npy"
    np.save(xp, np.zeros((5, 3)))
    assert main(["verify", str(sp), "--input", str(xp)]) == 2
    assert "state_dict" in capsys.readouterr().err
