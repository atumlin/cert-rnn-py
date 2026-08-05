# Certifying an LSTM Autoencoder Anomaly Detector on PMU Data

*How formal verification concepts (perturbations, frames, radii, specifications)
map onto the windowed-PMU anomaly-detection pipeline. Running example: a
2+2-layer LSTM-AE with hidden size H=55 on windows of shape (64, 1),
τ = 0.0545 chosen at the 95th percentile of validation scores.*

---

## 1. From PMU stream to model input: what a "window" is

The raw data is a time series: PMU measurements sampled at consecutive
timestamps. The autoencoder never sees the raw stream; it sees **sliding
windows** cut from it. One window is a matrix

> **x ∈ ℝ^(T×D)** with **T = 64** rows and **D = 1** column,

i.e., 64 consecutive timestamps of **one** measurement channel. The "weird
shape" (64, 1) is nothing deeper than that: row index = position in time
within the window, column index = which measured quantity (here there is
only one). The batch dimension (3256, 64, 1) exists only for training
throughput; verification always concerns **one window at a time**.

**Frames ≡ timestamps.** In the tool's vocabulary, "frame t" is row t of the
window — the measurement taken at the t-th timestamp of the window. There is
no other meaning. With D = 1, a frame holds a single scalar reading.

## 2. What the detector computes

The encoder compresses the window into a latent vector (the final top-layer
hidden state, ℝ^55); the decoder, fed that latent at every step, reproduces a
64-step reconstruction; a linear head maps each decoder state back to
measurement space. The anomaly score is the mean squared reconstruction error

> score(x) = (1/64) · Σₜ ( x̂ₜ − xₜ )²,

and the deployed rule is: **score(x) > τ ⇒ raise an anomaly alarm**. The
threshold τ = 0.0545 was set so that ~95% of normal validation windows fall
below it.

## 3. The specification: what property is being proven

Verification here does **not** ask "is the model accurate?" It asks a
robustness question about the *alarm decision* on a specific, known-normal
window (the **anchor**):

> **Spec (false-alarm robustness).** For every window x′ in a perturbation
> set B(x, ε) around the anchor x: score(x′) ≤ τ.

In words: *no perturbation within the set can trick the detector into a
false alarm on this normal window.* The proof covers **every** point of the
set — including the single worst one — not a sample of them. (The mirror-image
property, "no perturbation can mask a true anomaly below τ," is the same
machinery with the inequality flipped and an anomalous anchor; the shipped
spec is the false-alarm direction.)

## 4. What a perturbation is, concretely

A perturbation **changes measurement values, never timestamps**. Time
positions, window length, and sampling are fixed; what varies is the recorded
number at a timestamp. "Perturb frame t by up to ε" means: replace the reading
xₜ with any value in the interval [xₜ − ε, xₜ + ε]. Since D = 1, that is one
scalar per frame.

Physical readings of "the reading is off by at most ε": sensor noise and
calibration error, quantization, a transient communication glitch, or an
adversary with bounded ability to skew a measurement. **Units caveat:** ε
lives in the *normalized* units the model was trained on. To state a result
in physical units (volts, Hz, per-unit), multiply by the scale factor your
preprocessing applied to that channel.

Two threat models fix *which* frames may move:

- **single_frame** — exactly one timestamp's reading is corrupted; the other
  63 stay exactly as recorded. The tool certifies each frame separately and
  reports the per-frame radii and their minimum. Models: one bad sample, a
  single spoofed packet.
- **multi_frame** — all 64 readings are corrupted **simultaneously and
  independently**, each by up to ε, in the jointly worst way. Models: bounded
  noise on the whole channel, a sustained low-amplitude attack. Every
  single_frame set is contained in the multi_frame set, so the multi_frame
  radius is always the smaller of the two.

## 5. The certified radius, and why single_frame radii look huge

The **certified radius** is the largest ε for which the spec was *proven*
(found by bisection: try an ε, prove or fail, halve the step, repeat). Read
it as: "any perturbation smaller than this provably cannot cause a false
alarm on this window."

Expect a large asymmetry between the threat models, and do not be alarmed by
it. The score is a **mean over 64 values**, so a single corrupted sample is
diluted 64-fold. With τ = 0.0545 and anchor score 0.0209 the spare budget is
0.0336 in mean-squared terms — i.e. 64 × 0.0336 ≈ 2.15 of squared error
available to the one perturbed element, tolerating a reconstruction error of
≈ 1.47 there. A ±1 change in one normalized reading simply cannot move the
average past τ unless the network amplifies it across the window — and LSTMs
attenuate rather than amplify single-input influence. Hence single-frame
radii near or beyond 1 are *mathematically expected*, and a result of
0.999878 is literally the search's ceiling (1 − 2⁻¹³ with the default
bisection), meaning "everything tried was certified." The honest headline for
such a result: **no single corrupted sample within the data's realistic range
can cause a false alarm on this window.**

The **multi_frame radius is the discriminative number**: with all 64 readings
adversarially perturbed, errors add instead of dilute, and the radius lands
at a finite, informative value (typically orders of magnitude smaller). The
pair together tells the real story — robust to point corruption; quantified
sensitivity to coordinated or broadband perturbation.

## 6. What the proof engine does (one paragraph)

The tool pushes the entire *set* of perturbed windows through the network at
once, representing it as a zonotope (a center plus symbolic noise terms) and
soundly over-approximating each LSTM nonlinearity. Out the far end comes a
guaranteed **upper bound** on the score over the whole set; if that bound is
≤ τ, the spec is proven at that ε. Two consequences: (i) **soundness** — a
certificate is a real theorem about the model, covering the worst case;
(ii) **incompleteness** — a *failed* check does not prove an attack exists,
because the bound is conservative. The certified radius is therefore a
*floor* on the true robustness, never an overestimate of it.

## 7. Reporting checklist

A complete result for one window states: the anchor (which window, its
score), τ and how it was chosen, the threat model, the certified radius in
normalized *and* physical units, the bisection settings (they bound the
search range — report saturation as "≥ ceiling", not as the number), and
ideally a tightness estimate (certified bound vs. empirical worst case) so
readers can judge the conservatism. Template sentence:

> "For validation window #1424 (score 0.021, τ = 0.055), we certify that no
> simultaneous perturbation of all 64 measurements by less than ε* in
> normalized units (≙ … in physical units) can raise a false alarm
> (multi_frame). Under the single-measurement threat model the certificate
> saturates the search range (ε ≥ 1), i.e., no single corrupted reading can
> cause a false alarm."

---

## Condensed version

- **A window = 64 consecutive timestamps × 1 measurement channel.**
  Frame t = the reading at timestamp t. With D = 1, one frame = one scalar.
- **The property proven (Spec C):** for a chosen normal window, *no*
  perturbation within size ε can push the reconstruction score above τ —
  i.e., no false alarm, for every perturbation in the set, worst case included.
- **A perturbation changes values, not time:** reading xₜ may move anywhere
  in [xₜ−ε, xₜ+ε], in the model's normalized units.
- **single_frame:** one timestamp corrupted, rest exact; certified per frame;
  min over frames is the headline. **multi_frame:** all 64 corrupted at once;
  always a smaller radius; usually the informative one.
- **Certified radius:** the largest ε *proven* safe — a guaranteed floor on
  robustness (conservative, never optimistic).
- **Huge single_frame radii are expected, not a bug:** the score averages 64
  terms, so one sample is diluted 64×; 0.999878 is the bisection's ceiling
  (1 − 2⁻¹³), meaning "certified at everything tried." Report it as ε ≥ 1.
- **Report both radii + units translation:** point-corruption immunity
  (single_frame) and quantified noise tolerance (multi_frame), with ε mapped
  back to physical units via your normalization.
