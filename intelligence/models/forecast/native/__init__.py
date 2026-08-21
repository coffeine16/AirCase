"""Native (Numba/disk-streaming) execution path for train_and_promote,
opt-in via engine="native". The pandas/NumPy path in this package's
parent (intelligence/models/forecast/) remains the default and the
trusted reference -- see
docs/superpowers/specs/2026-08-21-local-native-training-pipeline-design.md
for the full rationale and the bit-for-bit parity bar every function here
is held to."""
