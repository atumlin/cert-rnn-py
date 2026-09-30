# Certifying an LSTM Autoencoder Anomaly Detector on PMU Data

*How formal verification concepts (perturbations, frames, radii,
specifications) map onto the windowed-PMU anomaly-detection pipeline of
`lstm_ae_model_example.ipynb`. Running example: a 2+2-layer LSTM-AE with
hidden size H=55 on inputs of shape (64, 1).*

> **Revision note.** An earlier version of this document assumed each input
> was 64 consecutive *timestamps* of one channel. Inspection of the actual
> data pipeline (`create_dataset`) shows otherwise — each input is one
> timestamp's vector of 64 *features*. Everything below reflects the real
> pipeline.

---

## 1. From CSV to model input: what an input "sequence" really is

The dataset rows are **per-timestamp snapshots**: each row of the zoneA CSV
holds the values of 64 SubFeature columns (PMU-derived quantities across the
zone) measured at one timestamp. The shaping code is:

```python
sequences = df.astype(np.float32).to_numpy().tolist()          # one element per ROW
dataset  = [torch.tensor(s).unsqueeze(1).float() for s in sequences]  # (64,) -> (64, 1)
```

So one model input **x ∈ ℝ^(64×1)** is a single row — one timestamp — with
its 64 feature values laid out along the "sequence" axis:

> **frame t = the t-th SubFeature column of one snapshot, NOT the t-th
> timestamp.** The "time" axis the LSTM iterates over is really the *feature
> index*; D = 1 because each "step" carries exactly one feature's scalar
> value.

The LSTM therefore reads the snapshot feature-by-feature in column order,
carrying a running 55-dimensional summary — a *feature-as-sequence*
autoencoder. This is a legitimate and common construction (the model learns
the joint structure/correlations of the feature vector; the fixed column
order plays the role time normally does), but it changes every domain
interpretation below. Temporal context across timestamps is **not** part of
one input; consecutive inputs are independent snapshots (the dataframe is
even shuffled before splitting).

Values are MinMax-scaled per column, with the scaler **fit on normal rows
only** — so each feature lives in ≈[0, 1], and each feature has its *own*
physical scale factor (see §6).

## 2. What the detector computes

Encoder (two stacked LSTMs, 55 units; the dropout module between them is
identity at inference) compresses the 64-feature snapshot into a latent
vector ℝ^55; the decoder receives that latent at every step and, through a
per-step linear head, reproduces a 64-value reconstruction x̂. The anomaly
score is

> score(x) = (1/64) · Σₜ ( x̂ₜ − xₜ )²  — mean squared error over the 64 features,

and the deployed rule is **score(x) > τ ⇒ anomaly alarm**. Note the deployed
threshold in the notebook is `THRESHOLD = np.percentile(train losses, 99)`.
(The verification cell recomputed a 95th-percentile-of-validation τ; for
certificates about the *deployed* detector, use the deployed `THRESHOLD` —
see §7.)

## 3. The specification being proven

For a chosen, known-normal snapshot (the **anchor**):

> **Spec (false-alarm robustness).** For every x′ in the perturbation set
> B(x, ε): score(x′) ≤ τ.

In words: *no allowed corruption of this snapshot's measurements can trick
the detector into a false alarm* — proven for **every** point of the set,
worst case included, not a sample.

## 4. What a perturbation is, in domain terms

A perturbation changes **measured values of the snapshot**, one per frame.
"Perturb frame t by up to ε" = replace SubFeature t's value with anything in
[xₜ − ε, xₜ + ε] (normalized units). The two threat models:

- **single_frame** — exactly **one SubFeature** of the snapshot is corrupted;
  the other 63 keep their recorded values. Domain reading: one bad sensor
  channel, one spoofed measurement, a single mis-registered quantity. The
  tool certifies each of the 64 features separately, yielding a **per-feature
  robustness profile** — which measured quantity most easily flips the alarm.
- **multi_frame** — **all 64 SubFeatures corrupted simultaneously**, each
  within ±ε, jointly worst-case. Domain reading: the entire measurement
  vector is noisy or adversarially skewed at once — bounded sensor noise
  across the zone, or a coordinated data-integrity attack on the snapshot.

Because a "window" is one timestamp, *both* threat models describe
corruption of a single instant's measurements; neither involves shifting or
perturbing anything across time.

## 5. The certified radius, and the single_frame dilution effect

The **certified radius** is the largest ε for which the spec was proven
(bisection: try, prove/fail, halve, repeat). Below it: provably no false
alarm. Above it: unknown (the method is conservative — the radius is a
*floor* on true robustness, never an overestimate).

Expect single_frame radii to be **large**, and don't read that as a bug. The
score averages 64 squared errors, so one corrupted feature is diluted 64×:
with τ = 0.0545 and anchor score 0.0209, the spare budget is
64 × 0.0336 ≈ 2.15 of squared error in the one perturbed element — tolerating
a reconstruction error of ≈1.47 there, on features that live in [0, 1]. A
certified value of 0.999878 is exactly the bisection's ceiling (1 − 2⁻¹³
with defaults), i.e. *every ε tried was certified*; report it as **ε ≥ 1**:
"no single corrupted SubFeature — over its entire physical range — can cause
a false alarm on this snapshot." A strong, true, and reportable property of
an averaging detector.

The **multi_frame radius is the discriminative number**: with all 64
features perturbed at once the errors add rather than dilute, and the radius
lands at a finite, informative value. Report the pair: immune to
single-channel corruption; quantified tolerance to vector-wide corruption.

## 6. Translating ε back to physical units

ε is uniform in *normalized* units, but MinMax scaling gives every feature
its own physical range, so one normalized ε means a different physical
magnitude per feature:

```python
phys_eps_per_feature = eps * scaler.data_range_   # array of 64 physical spans
```

For single_frame results this is natural (each frame's radius maps through
its own feature's range). For multi_frame, state it as "each SubFeature
simultaneously off by up to ε of its observed normal range."

## 7. Faithfulness of the verification to the deployed model

Three facts established by direct testing against the notebook's verbatim
legacy classes:

1. **Conversion parity holds.** The cert-rnn model reproduces the legacy
   model's reconstruction to ~1e-9 at batch size 1 (the preflight `torch
   parity` check). The dropout module is identity in eval mode and correctly
   absent from the converted model; the encoder's final rnn2 hidden state is
   exactly the latent the tool assumes; the decoder's `repeat` feeds that
   latent at every step for B=1, matching the tool's topology.
2. **B=1 is the deployed semantics.** The notebook's `predict()` scores
   samples one at a time, and the threshold was computed from B=1 losses —
   and verification is inherently B=1. (Observation, no action needed: the
   legacy decoder's `x.repeat(seq_len, n_features).reshape(...)` interleaves
   latents *across* samples when B>1, so batched training saw slightly
   scrambled decoder inputs. This affects what the weights are, not the
   correctness of certifying the resulting B=1 function.)
3. **τ alignment matters.** Certificates are statements "score stays ≤ τ."
   The deployed alarm uses `THRESHOLD` (99th percentile of train losses);
   a certificate against a different τ (e.g. 95th percentile of validation
   scores) is about a *different detector*. Use `tau=THRESHOLD` in the
   verification cell for deployment-relevant results.

---

## Condensed version

- **One input = one timestamp's snapshot of 64 SubFeatures** (one CSV row),
  reshaped to (64, 1). **Frame t = SubFeature t — features, not timestamps.**
  The LSTM reads the feature vector in column order as a pseudo-sequence.
- **Spec proven:** no perturbation of the snapshot's measurements within ε
  can push the reconstruction score above τ → no false alarm, worst case
  included.
- **single_frame** = one SubFeature corrupted (per-feature robustness
  profile). **multi_frame** = the whole 64-feature vector corrupted at once
  (the discriminative, smaller radius). Nothing temporal is perturbed.
- **Certified radius** = largest proven-safe ε; a conservative floor.
  Saturated single_frame results (0.999878 = the search ceiling, 1 − 2⁻¹³)
  mean "ε ≥ 1: no single feature, over its whole normal range, can cause a
  false alarm."
- **Units:** ε is in per-feature MinMax-normalized units; physical value =
  ε × `scaler.data_range_[j]`, different for each feature.
- **Tool faithfulness:** parity vs the verbatim legacy classes ≈1e-9 at B=1
  (deployment semantics); dropout correctly identity; use the deployed
  `THRESHOLD` (99th pct train), not a recomputed val percentile, as τ.
