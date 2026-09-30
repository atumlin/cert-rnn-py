# Zone-A LSTM-AE — end-to-end example notebooks

Full workflow on the Zone-A PMU SubFeatures data: load + normalize →
train an LSTM autoencoder → threshold evaluation → certification with
cert-rnn.

- `lstm_ae_model_example.ipynb` — v1: the original pipeline (one CSV row
  per sample, read feature-by-feature; 2-layer encoder/decoder stacks,
  H=55); verification via the baseline (box) certifier plus
  preflight/smoke-test and export to the overnight-script path.
- `lstm_ae_model_example_v2.ipynb` — v2: same model path, refined
  verification strategy (Cert-RNN+): ZRLT Tier 1 (`bilinear_mode("zono")`)
  vs the box baseline, the joint quadratic score bound, parallel/k-ary
  epsilon search, robustness curve, and per-frame radii. The heavy
  stages run OUTSIDE the notebook: the notebook saves a job file, the
  driver runs it detached, the notebook loads the JSON and draws figures.
- `lstm_ae_model_example_v3.ipynb` — v3: time-aware model. One sample is
  a window of 10 consecutive seconds × all SubFeatures, shape
  `(samples, 10, 64)`; files stay in time order, chronological 70/15/15
  split, scaler fit on the train block. **One code per second**: the
  encoder LSTM compresses each second's 64 features to 51 and decoder
  step t reads the encoder's hidden state at step t
  (`LSTMAutoencoder.from_torch(..., decoder_input="sequence")`). Adam,
  no weight decay. Verification with τ = the deployed THRESHOLD, Tier-1
  on every pass, attack selection (whole window / each second / chosen
  seconds), a fixed-ε check, batch windows (Option A: 10 back-to-back
  windows = one 100-second stretch; Option B: 100 windows across the test
  set) with a headline summary, and saved counterexamples.
- `verify_zoneA_v2.py` — the detached driver (run it inside tmux so an
  SSH drop costs nothing). Each stage is checkpointed to the results
  JSON as it finishes; rerunning skips completed stages. Main options
  (see its docstring):
  - `--attack whole|seconds|concentrated|all` and `--seconds 4,5,6`
  - `--eps 0.001,0.003` — fixed-ε check (VERIFIED / COUNTEREXAMPLE /
    UNKNOWN); counterexamples saved to `<out>_cex/*.npy`,
    `--minimize-cex` also saves the fewest-readings version
  - batch job files (`"anchors"` in the job) → per-window results plus
    a headline summary
  - `--zono-last-rounds`, `--multi-search`, `--tier1-threads`

Data (`SubFeatures_*_zoneA.csv`, `CyPhyScenarioLabels_*.csv`) is NOT in
this repo; run the notebooks where it lives. Outputs are stripped in the
committed copies. A trained checkpoint (`zoneA_ae_01_21_26.pth`) lets v2
skip straight to the verification section.
