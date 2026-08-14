"""Phase 0 instrumentation (proposal §10): M1/M2 recorders over an
instrumented LSTM step.

The engine is untouched. instrumented_lstm_step re-executes lstm.lstm_step's
exact body — same transformer calls, same order, same allocator discipline —
while recording per-gate operand-pair geometry. Parity with lstm_step is
asserted bit-for-bit in tests/soundness/test_instrument_parity.py; any drift
between the two bodies is a test failure, not a silent skew.

Recorded per gate application (layer, gate, t), per output coordinate k:
  cos        cosine similarity of the two operands' aligned generator rows
             (M2 proxy: correlation the box concretization discards)
  headroom   area(Z_k) / (w_x * w_y): fraction of the operand bounding box
             actually occupied by the joint 2-D zonotope Z_k. 1.0 = box is
             exact (nothing to recover); -> 0 = Z_k is a thin sliver (ZRLT
             Tier 1 headroom). Closed form, no vertex enumeration needed.
  w_x, w_y   operand interval widths
  fresh_w    width 2*|fresh generator| = (C2 - C1) added by this gate (M1)
  out_w      total output interval width (M1: fresh vs inherited split)

Plus a Z4 record per (layer, t): pairwise cosines among the cell-update
operand block (f_pre, i_pre, g_pre, c_prev) — the Tier 2 coupling quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cert_rnn.lstm import lstm_state_init
from cert_rnn.transformers import (
    bilinear_sigmoid_identity,
    bilinear_sigmoid_tanh,
)
from cert_rnn.zono import (
    PredAllocator,
    Zono,
    align_pred_space,
    zono_add,
)


# --------------------------------------------------------------------------- #
# geometry: closed-form 2-D zonotope area, O(p log p) per coordinate
# --------------------------------------------------------------------------- #
def zono2d_area(a: np.ndarray, b: np.ndarray) -> float:
    """Area of the 2-D zonotope with generators (a_i, b_i), alpha in [-1,1].

    Z is the Minkowski sum of segments [-g_i, +g_i] (segment vector 2 g_i),
    so area = sum_{i<j} |det[2 g_i, 2 g_j]| = 4 * sum_{i<j} |a_i b_j - a_j b_i|
    (axis-aligned check: generators (r_x, 0), (0, r_y) give 4 r_x r_y = w_x w_y).
    Canonicalizing every generator into the upper half-plane (angle in
    [0, pi)) makes every pairwise cross of angle-sorted generators
    non-negative, so the double sum collapses to prefix sums:
    sum_j cross(S_j, g_j), S_j = sum_{i<j} g_i.
    """
    flip = (b < 0) | ((b == 0) & (a < 0))
    a = np.where(flip, -a, a)
    b = np.where(flip, -b, b)
    order = np.argsort(np.arctan2(b, a))
    a = a[order]
    b = b[order]
    Sa = np.cumsum(a) - a
    Sb = np.cumsum(b) - b
    return 4.0 * float(np.sum(Sa * b - Sb * a))


def pair_geometry(z_x: Zono, z_y: Zono) -> dict:
    """Per-coordinate M2 geometry of an operand pair: cos, headroom, widths.

    Coordinates whose operands have zero generator mass get cos = nan and
    headroom = nan (there is no joint set to speak of).
    """
    _, (Vx, Vy) = align_pred_space(z_x, z_y)
    K = Vx.shape[0]
    cos = np.full(K, np.nan)
    headroom = np.full(K, np.nan)
    wx = 2.0 * np.sum(np.abs(Vx), axis=1)
    wy = 2.0 * np.sum(np.abs(Vy), axis=1)
    nx = np.linalg.norm(Vx, axis=1)
    ny = np.linalg.norm(Vy, axis=1)
    for k in range(K):
        if nx[k] > 0 and ny[k] > 0:
            cos[k] = float(Vx[k] @ Vy[k] / (nx[k] * ny[k]))
            headroom[k] = zono2d_area(Vx[k], Vy[k]) / (wx[k] * wy[k])
    return {"cos": cos, "headroom": headroom, "w_x": wx, "w_y": wy}


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #
@dataclass
class GateRecord:
    phase: str      # "enc" | "dec"
    t: int
    layer: int
    gate: str       # "f*c_prev" | "i*g" | "o*tanh(c)"
    cos: np.ndarray
    headroom: np.ndarray
    w_x: np.ndarray
    w_y: np.ndarray
    fresh_w: np.ndarray
    out_w: np.ndarray


@dataclass
class Z4Record:
    phase: str
    t: int
    layer: int
    # pairwise cosines among (f_pre, i_pre, g_pre, c_prev), keys "f:i" etc.
    cos: dict


@dataclass
class StepRecorder:
    gates: list = field(default_factory=list)
    z4: list = field(default_factory=list)
    phase: str = "enc"

    def record_gate(self, phase, t, layer, gate, z_x, z_y, z_out):
        geo = pair_geometry(z_x, z_y)
        operand_ids = set(z_x.pred_ids) | set(z_y.pred_ids)
        fresh_cols = [i for i, pid in enumerate(z_out.pred_ids)
                      if pid not in operand_ids]
        fresh_w = 2.0 * np.sum(np.abs(z_out.V[:, fresh_cols]), axis=1) \
            if fresh_cols else np.zeros(z_out.dim)
        out_w = 2.0 * np.sum(np.abs(z_out.V), axis=1)
        self.gates.append(GateRecord(phase, t, layer, gate,
                                     geo["cos"], geo["headroom"],
                                     geo["w_x"], geo["w_y"], fresh_w, out_w))

    def record_z4(self, phase, t, layer, z_f, z_i, z_g, z_c_prev):
        named = {"f": z_f, "i": z_i, "g": z_g, "c": z_c_prev}
        keys = list(named)
        cos = {}
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                geo = pair_geometry(named[keys[a]], named[keys[b]])
                cos[f"{keys[a]}:{keys[b]}"] = geo["cos"]
        self.z4.append(Z4Record(phase, t, layer, cos))


# --------------------------------------------------------------------------- #
# instrumented step / reach (bodies mirror lstm.py / verify.py exactly)
# --------------------------------------------------------------------------- #
def instrumented_lstm_step(z_x, z_h_prev, z_c_prev, W_in, W_rec, b,
                           allocator: PredAllocator,
                           rec: StepRecorder, t: int, layer: int):
    """lstm.lstm_step body + recording. Must stay call-for-call identical."""
    H = W_in.shape[0] // 4
    z_in_proj = z_x.affine_map(W_in, b)
    z_rec_proj = z_h_prev.affine_map(W_rec, None)
    z_pre = zono_add(z_in_proj, z_rec_proj)

    z_i_pre = z_pre.slice_rows(0, H)
    z_f_pre = z_pre.slice_rows(H, 2 * H)
    z_g_pre = z_pre.slice_rows(2 * H, 3 * H)
    z_o_pre = z_pre.slice_rows(3 * H, 4 * H)

    z_c_term1 = bilinear_sigmoid_identity(z_c_prev, z_f_pre, allocator)
    z_c_term2 = bilinear_sigmoid_tanh(z_i_pre, z_g_pre, allocator)
    z_c = zono_add(z_c_term1, z_c_term2)
    z_h = bilinear_sigmoid_tanh(z_o_pre, z_c, allocator)

    rec.record_gate(rec.phase, t, layer, "f*c_prev", z_c_prev, z_f_pre, z_c_term1)
    rec.record_gate(rec.phase, t, layer, "i*g", z_i_pre, z_g_pre, z_c_term2)
    rec.record_gate(rec.phase, t, layer, "o*tanh(c)", z_o_pre, z_c, z_h)
    rec.record_z4(rec.phase, t, layer, z_f_pre, z_i_pre, z_g_pre, z_c_prev)
    return z_h, z_c


def instrumented_lstm_step_stack(z_x, z_h_layers, z_c_layers, layers,
                                 allocator, rec, t):
    new_h = [None] * len(layers)
    new_c = [None] * len(layers)
    inp = z_x
    for i, lyr in enumerate(layers):
        new_h[i], new_c[i] = instrumented_lstm_step(
            inp, z_h_layers[i], z_c_layers[i],
            lyr["W_in"], lyr["W_rec"], lyr["b"], allocator, rec, t, i)
        inp = new_h[i]
    return new_h, new_c


def instrumented_lstm_ae_reach(encoder, decoder, head, x_anchor, eps,
                               threat_model="multi_frame", t_pert=None,
                               allocator: PredAllocator | None = None):
    """verify.lstm_ae_reach body + recording. Returns
    (z_x_hat_seq, z_x_seq, recorder)."""
    from cert_rnn.verify import _build_input_zonos
    from cert_rnn.zono import get_default_allocator

    # Same allocator discipline as verify.lstm_ae_reach: the default
    # module allocator issues BOTH the input-zono ids (inside
    # _build_input_zonos) and the fresh transformer ids, so id spaces
    # cannot collide. A custom allocator here would alias fresh ids with
    # input ids.
    alloc = allocator if allocator is not None else get_default_allocator()
    rec = StepRecorder()
    H = encoder["H"]
    L_enc, L_dec = encoder["L"], decoder["L"]
    T, _D = x_anchor.shape
    z_x_seq = _build_input_zonos(x_anchor, eps, threat_model, t_pert)

    rec.phase = "enc"
    z_h_enc, z_c_enc = lstm_state_init(H, L_enc)
    for t in range(T):
        z_h_enc, z_c_enc = instrumented_lstm_step_stack(
            z_x_seq[t], z_h_enc, z_c_enc, encoder["layers"], alloc, rec, t)
    z_latent = z_h_enc[-1]

    rec.phase = "dec"
    z_h_dec, z_c_dec = lstm_state_init(H, L_dec)
    z_x_hat_seq = []
    for t in range(T):
        z_h_dec, z_c_dec = instrumented_lstm_step_stack(
            z_latent, z_h_dec, z_c_dec, decoder["layers"], alloc, rec, t)
        z_x_hat_seq.append(z_h_dec[-1].affine_map(head["W"], head["b"]))
    return z_x_hat_seq, z_x_seq, rec
