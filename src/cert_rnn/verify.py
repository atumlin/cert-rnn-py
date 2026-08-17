"""Cert-RNN verification: Algorithm 1 bisection, reach-set, Spec A, Spec C.

Two specs:
  - Spec A (classifier margin): certify that logit[true_class] dominates
    every other logit over the entire eps-ball -- i.e., argmax does not
    change. Sound via componentwise diff bbox.
  - Spec C (autoencoder false-alarm): certify that
        score(x') := ||AE(x') - x'||_2^2 / N <= tau
    for every x' in the eps-ball, where AE is encoder+decoder+per-step
    linear head. Sound upper bound:
        score_ub = (1/N) * sum_{t,d} (|c_diff[t,d]| + sum_p |V_diff[t,d,p]|)^2
    componentwise worst-case squared, summed. Over-bounds
    max-of-sum-of-squares; tighter joint bounds need a quadratic-over-box
    solver (out of scope).

Threat models:
  - 'single_frame' (Algorithm 1): one frame perturbed by eps; pin the
    others. The sample's certified radius is min over frames.
  - 'multi_frame': every frame perturbed independently with disjoint
    pred_ids. Sound under this port's Minkowski-padded lstm_step; the
    MATLAB reference is unsound here.

Algorithm 1 (Du et al., CCS 2021): start at eps_init; at iteration l
in [2, n_iters+1], add 0.5^l if certify holds at eps, else subtract
0.5^l. Track the largest eps that ever certified.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from cert_rnn.lstm import lstm_state_init, lstm_step_stack
from cert_rnn.rnn import rnn_step
from cert_rnn.zono import Zono, zono_sub

ThreatModel = Literal["single_frame", "multi_frame"]


# ---------- Algorithm 1 ----------


def bisect_epsilon(
    certify_fn: Callable[[float], bool],
    eps_init: float = 0.5,
    n_iters: int = 12,
) -> float:
    """Du et al. Algorithm 1 bisection on epsilon.

    Returns the largest eps for which certify_fn(eps) returned True.
    Returns 0.0 if no eps in the search trajectory certified.
    """
    eps = eps_init
    best = 0.0
    for ell in range(2, n_iters + 2):
        eps = max(eps, 0.0)
        ok = certify_fn(eps)
        if ok:
            best = max(best, eps)
            eps = eps + 0.5 ** ell
        else:
            eps = eps - 0.5 ** ell
    if eps > 0 and certify_fn(eps):
        best = max(best, eps)
    return best


# ---------- input zono construction ----------


def _build_input_zonos(
    x_seq: np.ndarray, eps: float, threat_model: ThreatModel, t_pert: int | None
) -> list[Zono]:
    T, D = x_seq.shape
    if threat_model == "single_frame":
        if t_pert is None:
            raise ValueError("single_frame requires t_pert")
        if not (0 <= t_pert < T):
            raise ValueError(f"t_pert {t_pert} out of range [0, {T})")
        out = []
        for t in range(T):
            if t == t_pert and eps > 0:
                out.append(Zono.from_box(x_seq[t], eps))
            else:
                out.append(Zono.point(x_seq[t]))
        return out
    if threat_model == "multi_frame":
        return [
            Zono.from_box(x_seq[t], eps) if eps > 0 else Zono.point(x_seq[t])
            for t in range(T)
        ]
    raise ValueError(f"unknown threat_model {threat_model!r}")


# ---------- reach-set ----------


def lstm_reach(
    model_dict: dict,
    x_seq: np.ndarray,
    eps: float,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> list[Zono]:
    """Forward an LSTM-stack model over an eps-perturbed input sequence.

    Returns the per-timestep top-layer hidden zonotope list. If
    model_dict has a 'head', the classifier head is NOT applied here --
    callers compose it via Zono.affine_map.
    """
    H, L = model_dict["H"], model_dict["L"]
    z_x_seq = _build_input_zonos(x_seq, eps, threat_model, t_pert)
    z_h, z_c = lstm_state_init(H, L)
    z_h_top_seq: list[Zono] = []
    for t in range(x_seq.shape[0]):
        z_h, z_c = lstm_step_stack(z_x_seq[t], z_h, z_c, model_dict["layers"])
        z_h_top_seq.append(z_h[-1])
    return z_h_top_seq


def lstm_ae_reach(
    encoder: dict,
    decoder: dict,
    head: dict,
    x_anchor: np.ndarray,
    eps: float,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> tuple[list[Zono], list[Zono]]:
    """Forward an LSTM autoencoder: encoder over x_anchor, decoder reads
    the latent (encoder's final top-layer h) at every step, per-step
    head produces a reconstruction zono per timestep.

    Returns (z_x_hat_seq, z_x_seq) for downstream spec_c_score_ub.
    """
    H = encoder["H"]
    L_enc, L_dec = encoder["L"], decoder["L"]
    T, D = x_anchor.shape
    z_x_seq = _build_input_zonos(x_anchor, eps, threat_model, t_pert)

    z_h_enc, z_c_enc = lstm_state_init(H, L_enc)
    for t in range(T):
        z_h_enc, z_c_enc = lstm_step_stack(
            z_x_seq[t], z_h_enc, z_c_enc, encoder["layers"]
        )
    z_latent = z_h_enc[-1]

    z_h_dec, z_c_dec = lstm_state_init(H, L_dec)
    z_x_hat_seq: list[Zono] = []
    for _t in range(T):
        z_h_dec, z_c_dec = lstm_step_stack(
            z_latent, z_h_dec, z_c_dec, decoder["layers"]
        )
        z_x_hat_seq.append(z_h_dec[-1].affine_map(head["W"], head["b"]))
    return z_x_hat_seq, z_x_seq


# ---------- specs ----------


def spec_a_margin(
    model_dict: dict,
    x_seq: np.ndarray,
    eps: float,
    true_class: int,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> bool:
    """Certify: logit[true_class] > logit[c] for every c != true_class
    over the entire eps-ball.

    Sound check: build the (C-1, C) difference matrix D where
        D[i, true_class] = +1, D[i, c_i] = -1
    apply it to the logits zono, and assert the resulting bbox has lb > 0
    on every row.
    """
    if "head" not in model_dict:
        raise ValueError("spec_a_margin requires model_dict['head']")
    z_h_top_seq = lstm_reach(model_dict, x_seq, eps, threat_model, t_pert)
    z_h_T = z_h_top_seq[-1]
    z_logits = z_h_T.affine_map(model_dict["head"]["W"], model_dict["head"]["b"])
    C = z_logits.dim
    if not (0 <= true_class < C):
        raise ValueError(f"true_class {true_class} out of range [0, {C})")
    others = [c for c in range(C) if c != true_class]
    diffs = np.zeros((len(others), C), dtype=np.float64)
    for i, c in enumerate(others):
        diffs[i, true_class] = 1.0
        diffs[i, c] = -1.0
    z_diff = z_logits.affine_map(diffs)
    lb, _ = z_diff.get_ranges()
    return bool(np.all(lb > 0))


def spec_c_score_ub(z_x_hat_seq: list[Zono], z_x_seq: list[Zono]) -> float:
    """Sound upper bound on score(x') = ||AE(x') - x'||_2^2 / N over the
    perturbation set defined by z_x_seq.

        score_ub = (1/N) * sum_{t, d} (|c_diff[t,d]| + sum_p |V_diff[t,d,p]|)^2

    Componentwise worst-case |diff| squared, summed. Over-bounds
    max-of-sum-of-squares; tighter joint bounds would need a
    quadratic-over-box solver.
    """
    if len(z_x_hat_seq) != len(z_x_seq):
        raise ValueError("z_x_hat_seq and z_x_seq must have the same length")
    T = len(z_x_hat_seq)
    if T == 0:
        return 0.0
    D = z_x_hat_seq[0].dim
    N = T * D
    score_ub = 0.0
    for t in range(T):
        z_diff = zono_sub(z_x_hat_seq[t], z_x_seq[t])
        radius = np.sum(np.abs(z_diff.V), axis=1)
        comp_max = np.abs(z_diff.c) + radius
        score_ub += float(np.sum(comp_max ** 2))
    return score_ub / N


def _stack_residuals(z_x_hat_seq: list[Zono], z_x_seq: list[Zono]):
    """Residual zonotopes AE(x')-x' per step, stacked into the union
    predicate space. Returns (c, V) with c: (N,), V: (N, P)."""
    if len(z_x_hat_seq) != len(z_x_seq):
        raise ValueError("z_x_hat_seq and z_x_seq must have the same length")
    diffs = [zono_sub(a, b) for a, b in zip(z_x_hat_seq, z_x_seq)]
    all_ids: dict = {}
    for d in diffs:
        for pid in d.pred_ids:
            if pid not in all_ids:
                all_ids[pid] = len(all_ids)
    N = sum(d.dim for d in diffs)
    c = np.concatenate([d.c for d in diffs])
    V = np.zeros((N, len(all_ids)))
    row = 0
    for d in diffs:
        V[row:row + d.dim, [all_ids[p] for p in d.pred_ids]] = d.V
        row += d.dim
    return c, V


def spec_c_score_ub_joint(z_x_hat_seq: list[Zono], z_x_seq: list[Zono]) -> float:
    """Tighter sound upper bound on score(x') over the perturbation set,
    exploiting that every residual component shares the SAME generator
    vector alpha:

        max_{alpha in [-1,1]^P} ||c + V alpha||^2
          <= ||c||^2 + 2 ||V^T c||_1 + sum_{ij} |(V^T V)_{ij}|

    Provably <= the componentwise bound of spec_c_score_ub (push the
    absolute values inside both inner products to recover it), strictly
    tighter whenever cancellation exists across components. Cost: one
    (P x N) @ (N x P) Gram product -- grows with the generator count P,
    so the componentwise bound remains the cheap default; `certify` /
    suite adapters select via score_bound="joint".
    """
    T = len(z_x_hat_seq)
    if T == 0:
        return 0.0
    N = T * z_x_hat_seq[0].dim
    c, V = _stack_residuals(z_x_hat_seq, z_x_seq)
    M = V.T @ V
    ub = float(c @ c) + 2.0 * float(np.abs(V.T @ c).sum()) + float(np.abs(M).sum())
    # The componentwise bound can win only by fp noise; take the min --
    # both are sound.
    return min(ub / N, spec_c_score_ub(z_x_hat_seq, z_x_seq))


def spec_c_score_lb_joint(z_x_hat_seq: list[Zono], z_x_seq: list[Zono]) -> float:
    """Tighter sound LOWER bound on score(x') over the perturbation set
    (the masking property's certificate):

        min_alpha ||c + V alpha||_2 >= ||c||_2 - max_alpha ||V alpha||_2
                                    >= ||c||_2 - sqrt(sum_{ij} |(V^T V)_{ij}|)

    combined (max) with the componentwise lower bound -- neither
    dominates, both are sound."""
    T = len(z_x_hat_seq)
    if T == 0:
        return 0.0
    N = T * z_x_hat_seq[0].dim
    c, V = _stack_residuals(z_x_hat_seq, z_x_seq)
    M = V.T @ V
    norm_lb = max(0.0, float(np.linalg.norm(c)) - float(np.sqrt(np.abs(M).sum())))
    # componentwise lower bound: per component |diff_d| >= max(0, lb, -ub)
    radius = np.abs(V).sum(axis=1)
    comp_min = np.maximum(0.0, np.maximum(c - radius, -(c + radius)))
    comp_lb = float(np.sum(comp_min ** 2))
    return max(norm_lb ** 2, comp_lb) / N


def spec_c_holds(
    encoder: dict,
    decoder: dict,
    head: dict,
    x_anchor: np.ndarray,
    eps: float,
    tau: float,
    threat_model: ThreatModel = "single_frame",
    t_pert: int | None = None,
) -> bool:
    """Spec C wrapper: True iff sound score_ub <= tau."""
    z_xh, z_x = lstm_ae_reach(
        encoder, decoder, head, x_anchor, eps, threat_model, t_pert
    )
    return spec_c_score_ub(z_xh, z_x) <= tau


# ---------- k-ary epsilon search (Phase 1 throughput, 2b) ----------


def _kary_rounds(resolution_bits: int, probes: int) -> list[int]:
    """Bits per round for a k-ary search with `probes` = 2^a - 1 probes per
    round that must end EXACTLY on the 2^-resolution_bits grid: rounds of a
    bits with a final round carrying the remainder. probes=15 (a=4) and
    13 bits -> [4, 4, 4, 1]: four sequential rounds (46 probes) — three
    rounds would stop at 2^-12 and could return a coarser radius."""
    a = int(round(np.log2(probes + 1)))
    if 2 ** a - 1 != probes:
        raise ValueError(f"probes must be 2^a - 1, got {probes}")
    rounds = []
    left = resolution_bits
    while left > 0:
        b = min(a, left)
        rounds.append(b)
        left -= b
    return rounds


def kary_epsilon(
    certify_many: Callable[[list], list],
    eps_init: float = 0.5,
    n_iters: int = 12,
    probes: int = 15,
) -> tuple[float, int]:
    """k-ary replacement for bisect_epsilon on the SAME grid.

    bisect_epsilon(eps_init, n_iters) is bisection on the open interval
    (eps_init - 0.5, eps_init + 0.5) with n_iters + 1 probes, i.e. it
    returns the largest certified point of the grid
        G = { eps_init - 0.5 + j * 2^-(n_iters+1) : j = 1 .. 2^(n_iters+1) - 1 }
    (for a monotone oracle). This function narrows to the same grid cell
    with `probes` parallel probes per round: each round subdivides the
    current cell into probes+1 equal parts (a power of two so the grid is
    preserved), keeping the sub-cell whose lower end is the largest
    certified probe. Returns (largest certified eps, number of rounds).

    certify_many(list_of_eps) -> list_of_bool evaluates probes (in
    parallel if the caller wishes). Requires eps_init >= 0.5, where
    Algorithm 1's clamp `eps = max(eps, 0)` never fires and the two searches
    are provably on the same grid.
    """
    if eps_init < 0.5:
        raise ValueError("kary_epsilon requires eps_init >= 0.5 (grid equivalence)")
    bits = n_iters + 1
    lo = eps_init - 0.5          # virtual certified end (never probed)
    hi = eps_init + 0.5          # virtual failed end (never probed)
    best = 0.0
    n_rounds = 0
    for b in _kary_rounds(bits, probes):
        m = 2 ** b
        step = (hi - lo) / m
        eps_list = [lo + j * step for j in range(1, m)]
        oks = certify_many(eps_list)
        n_rounds += 1
        new_lo, new_hi = lo, hi
        for e, ok in zip(eps_list, oks):
            if ok:
                new_lo = e
                best = max(best, e)
        for e in eps_list:
            if e > new_lo:
                new_hi = e
                break
        lo, hi = new_lo, new_hi
    return best, n_rounds


# ---------- parallel per-frame driver (Phase 1 throughput, 2a) ----------

_WORKER_PAYLOAD: dict = {}


def _init_worker(payload: dict) -> None:
    """Process-pool initializer: stash the (small) model payload once per
    worker so jobs carry only (frame, eps). Each worker has its own default
    predicate allocator; the absolute id offset does not affect results
    because every zonotope's columns are ordered by allocation order within
    a reach (ids are monotone), not by absolute id."""
    from cert_rnn.transformers import set_bilinear_mode

    _WORKER_PAYLOAD.clear()
    _WORKER_PAYLOAD.update(payload)
    set_bilinear_mode(payload.get("bilinear_mode", "box"))
    from cert_rnn.transformers import set_tilt_mode
    set_tilt_mode(payload.get("tilt_mode", "cornerfit"))


def _worker_probe(job):
    """job = (frame_or_None, eps) or (frame_or_None, eps, mode_override).
    mode_override temporarily forces the bilinear mode for this probe
    (used by the Tier-1 round-cutover schedule)."""
    from cert_rnn.transformers import bilinear_mode

    P = _WORKER_PAYLOAD
    t, eps = job[0], job[1]
    mode = job[2] if len(job) > 2 else None

    def probe():
        if P["spec"] == "c":
            return spec_c_holds(P["encoder"], P["decoder"], P["head"], P["x"],
                                eps, P["tau"], P["threat_model"], t)
        return spec_a_margin(P["model_dict"], P["x"], eps, P["true_class"],
                             P["threat_model"], t)

    if mode is None:
        return probe()
    with bilinear_mode(mode):
        return probe()


def _search_frames(
    payload: dict,
    frames: list,
    eps_init: float,
    n_iters: int,
    search: str,
    probes: int,
    n_workers: int,
    zono_last_rounds: int | None = None,
) -> tuple[np.ndarray, int]:
    """Run the epsilon search for every frame in `frames` (or [None] for
    multi_frame). search: "bisect" (Algorithm 1) or "kary". n_workers > 1
    evaluates probes in a process pool; the serial path runs the SAME
    lockstep walk through a serial mapper, so the two are identical by
    construction (and asserted bit-identical by the regression tests).

    zono_last_rounds (Tier-1 cutover, sound-but-may-lose-tightness): when
    the session bilinear mode is "zono", probes in all but the LAST
    `zono_last_rounds` rounds are evaluated in "box" mode. Both modes are
    sound, so any mixed schedule certifies only true radii; early box
    rejections can steer the walk lower, so the result is >= the all-box
    radius and <= the all-zono radius. None = current mode everywhere.

    Returns (per-frame radii, sequential rounds)."""
    from cert_rnn.transformers import get_bilinear_mode, get_tilt_mode

    session_mode = get_bilinear_mode()
    payload = dict(payload, bilinear_mode=session_mode, tilt_mode=get_tilt_mode())
    if search not in ("bisect", "kary"):
        raise ValueError(f"unknown search {search!r}")

    if search == "bisect":
        total_rounds = n_iters + 1
    else:
        if eps_init < 0.5:
            raise ValueError("kary requires eps_init >= 0.5")
        total_rounds = len(_kary_rounds(n_iters + 1, probes))

    def mode_for_round(r):   # r is 1-based
        if (zono_last_rounds is None or session_mode != "zono"
                or r > total_rounds - zono_last_rounds):
            return None      # payload default
        return "box"

    def walk(mapper):
        if search == "bisect":
            state = [{"eps": eps_init, "best": 0.0} for _ in frames]
            r = 0
            for ell in range(2, n_iters + 2):
                r += 1
                m = mode_for_round(r)
                jobs = [(t, max(st["eps"], 0.0), m)
                        for t, st in zip(frames, state)]
                oks = mapper(jobs)
                for st, jb, ok in zip(state, jobs, oks):
                    e = jb[1]
                    if ok:
                        st["best"] = max(st["best"], e)
                        st["eps"] = e + 0.5 ** ell
                    else:
                        st["eps"] = e - 0.5 ** ell
            r += 1
            m = mode_for_round(r)
            jobs = [(t, st["eps"], m) for t, st in zip(frames, state)]
            oks = mapper(jobs)
            for st, jb, ok in zip(state, jobs, oks):
                if jb[1] > 0 and ok:
                    st["best"] = max(st["best"], jb[1])
            return np.array([st["best"] for st in state]), total_rounds
        # k-ary, lockstep across frames
        bits = n_iters + 1
        state = [{"lo": eps_init - 0.5, "hi": eps_init + 0.5, "best": 0.0}
                 for _ in frames]
        r = 0
        for b in _kary_rounds(bits, probes):
            r += 1
            mmode = mode_for_round(r)
            m = 2 ** b
            jobs = []
            owner = []
            for i, (t, st) in enumerate(zip(frames, state)):
                step = (st["hi"] - st["lo"]) / m
                for j in range(1, m):
                    jobs.append((t, st["lo"] + j * step, mmode))
                    owner.append(i)
            oks = mapper(jobs)
            per = {}
            for jb, o, ok in zip(jobs, owner, oks):
                per.setdefault(o, []).append((jb[1], ok))
            for i, st in enumerate(state):
                probes_i = per.get(i, [])
                new_lo, new_hi = st["lo"], st["hi"]
                for e, ok in probes_i:
                    if ok:
                        new_lo = e
                        st["best"] = max(st["best"], e)
                for e, _ in probes_i:
                    if e > new_lo:
                        new_hi = e
                        break
                st["lo"], st["hi"] = new_lo, new_hi
        return np.array([st["best"] for st in state]), r

    if n_workers <= 1:
        _init_worker(payload)
        return walk(lambda jobs: [_worker_probe(j) for j in jobs])

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(payload,)) as ex:
        return walk(lambda jobs: list(
            ex.map(_worker_probe, jobs,
                   chunksize=max(1, len(jobs) // (4 * n_workers)))))


# ---------- certified radius (bisection over eps) ----------


def certify_radius_spec_a(
    model_dict: dict,
    x_seq: np.ndarray,
    true_class: int,
    eps_init: float = 0.5,
    n_iters: int = 12,
    threat_model: ThreatModel = "single_frame",
    search: str = "bisect",
    probes: int = 15,
    n_workers: int = 1,
    zono_last_rounds: int | None = None,
) -> tuple[float, np.ndarray | None]:
    """Epsilon search for Spec A.

    single_frame: search per frame; return (min over frames, per-frame array).
    multi_frame:  search once; return (eps, None).
    search: "bisect" (Algorithm 1, n_iters + 1 sequential probes) or "kary"
    (`probes` per round on the same 2^-(n_iters+1) grid — identical result,
    fewer sequential rounds). n_workers > 1 evaluates probes (and frames)
    in a process pool; results are bit-identical to the sequential path.
    """
    payload = {"spec": "a", "model_dict": model_dict, "x": x_seq,
               "true_class": true_class, "threat_model": threat_model}
    if threat_model == "single_frame":
        frames = list(range(x_seq.shape[0]))
        per_frame, _ = _search_frames(payload, frames, eps_init, n_iters,
                                      search, probes, n_workers, zono_last_rounds)
        return float(per_frame.min()), per_frame
    r, _ = _search_frames(payload, [None], eps_init, n_iters, search, probes,
                          n_workers, zono_last_rounds)
    return float(r[0]), None


def certify_radius_spec_c(
    encoder: dict,
    decoder: dict,
    head: dict,
    x_anchor: np.ndarray,
    tau: float,
    eps_init: float = 0.5,
    n_iters: int = 12,
    threat_model: ThreatModel = "single_frame",
    search: str = "bisect",
    probes: int = 15,
    n_workers: int = 1,
    zono_last_rounds: int | None = None,
) -> tuple[float, np.ndarray | None]:
    """Epsilon search for Spec C. Same shape/options as certify_radius_spec_a."""
    payload = {"spec": "c", "encoder": encoder, "decoder": decoder, "head": head,
               "x": x_anchor, "tau": tau, "threat_model": threat_model}
    if threat_model == "single_frame":
        frames = list(range(x_anchor.shape[0]))
        per_frame, _ = _search_frames(payload, frames, eps_init, n_iters,
                                      search, probes, n_workers, zono_last_rounds)
        return float(per_frame.min()), per_frame
    r, _ = _search_frames(payload, [None], eps_init, n_iters, search, probes,
                          n_workers, zono_last_rounds)
    return float(r[0]), None
