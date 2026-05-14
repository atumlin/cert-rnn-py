"""PyTorch interop: extract Cert-RNN model dicts from nn.LSTM / nn.RNN.

Stub for Phase 0. Lands in Phase 2.

PyTorch gate order for nn.LSTM weight_ih_l*/weight_hh_l* is
[i, f, g, o] -- same as the Cert-RNN engine; no row permutation needed.
PyTorch carries two biases (bias_ih, bias_hh); they are summed into a
single b to match the engine signature.

Adapted from cert_rnn_export.py in the sibling MATLAB reference repo;
this port drops the scipy.io.savemat boundary -- model dicts stay in
Python.
"""
