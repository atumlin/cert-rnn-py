# Zone-A LSTM-AE — end-to-end example notebooks

Full workflow on the Zone-A PMU SubFeatures data: load + normalize →
train an LSTM autoencoder (2-layer encoder/decoder stacks, H=16) →
threshold evaluation → certification with cert-rnn.

- `lstm_ae_model_example.ipynb` — v1: the original pipeline; verification
  via the baseline (box) certifier plus preflight/smoke-test and export
  to the overnight-script path.
- `lstm_ae_model_example_v2.ipynb` — v2: same model path, refined
  verification strategy (Cert-RNN+): ZRLT Tier 1 (`bilinear_mode("zono")`)
  vs the box baseline side by side, the joint quadratic score bound,
  parallel/k-ary epsilon search, robustness curve, and per-frame radii.

Data (`SubFeatures_*_zoneA.csv`, `CyPhyScenarioLabels_*.csv`) is NOT in
this repo; run the notebooks where it lives. Outputs are stripped in the
committed copies. A trained checkpoint (`zoneA_ae_01_21_26.pth`) lets v2
skip straight to the verification section.
