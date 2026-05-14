"""Verification: Algorithm 1 bisection, reach-set, spec_a and spec_c.

Stub for Phase 0. Lands in Phase 3.

Will implement:
    bisect_epsilon(certify_fn, eps_init, n_iters)   Du et al. Algorithm 1.
    spec_a_classifier_margin(model, x, eps, ...)    correct-class robust margin.
    spec_c_score_ub(model, x_anchor, eps, ...)      LSTM-AE false-alarm bound.
    reach_set(model, x, eps, ...)                   per-step output Zonos.

Threat models:
    'single_frame'  (Algorithm 1 default): perturb one frame at a time,
                    pin the rest.
    'multi_frame'   full L_inf on the input sequence; sound only with
                    Minkowski-padded lstm_step (see cert_rnn.lstm).
"""
