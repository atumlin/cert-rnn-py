# Soundness

Stub. Lands alongside the engine in Phase 1. Will document:

- Zonotope abstract domain and predicate semantics.
- Parallel-tangent bounding planes for tanh and sigmoid (§4.2.2).
- Corner-fit (A, B) for the two bilinear transformers; exact (C1, C2)
  via corner + edge-stationary + (for sigma·tanh) quartic-root search.
- Minkowski-padded LSTM step: why disjoint predicate slots restore
  soundness for fresh-per-timestep perturbation, and why the
  shared-prefix scheme in the MATLAB reference fails on that pattern.
- Spec C aggregation: componentwise worst-case squares are a sound
  upper bound on score(x') = ||AE(x') - x'||_2^2 / N.

Citations:
  Du, T., Ji, S., Shen, L., Zhang, Y., Li, J., Shi, J., Fang, C.,
  Yin, J., Beyah, R., Wang, T. (2021). "Cert-RNN: Towards Certifying
  the Robustness of Recurrent Neural Networks." CCS '21.
