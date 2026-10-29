# Models and inference libraries

**Rule: no non-commercial weights.** MusicDecoded has paid plans, so every model we run must have a
**code license and a weights license** that allow commercial use. Check the weights, not just the
code (PLAN.md R-7). Add a row here before adding any model.

| Component | Used for | Where | Code license | Weights license | Tier |
|---|---|---|---|---|---|
| [librosa](https://github.com/librosa/librosa) | Beat tracking, audio loading, features | `services/librosa_detector_service.py` | ISC | n/a (no learned weights) | Free (standard path) |
| [Beat Transformer](https://github.com/zhaojw1998/Beat-Transformer) | Beat + downbeat detection | `models/Beat-Transformer` (submodule), `models/beat_transformer.py`, `services/beat_transformer_service.py` | MIT | MIT (checkpoints ship in the upstream repo under its MIT license) | Pro (premium path) |
| [ISMIR2019 large-vocabulary chord recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) | Chord recognition (CNN-LSTM) | `models/chord-recognition-model` (submodule), `services/chord_cnn_lstm_service.py` | MIT | MIT (pretrained models ship in the upstream repo under its MIT license) | Free + Pro |

Supporting libraries without learned weights: PyTorch (BSD-3), NumPy (BSD-3), SciPy (BSD-3, `find_peaks`
in the Beat Transformer wrapper).

## Not allowed

| Component | Why |
|---|---|
| madmom pretrained models (`RNNBeatProcessor`, `RNNDownBeatProcessor`, etc.) | CC BY-NC-SA 4.0 (non-commercial). The madmom source is BSD, but the `madmom` pip package bundles the NC models, so it is not installed at all. Removed in T-116. |

## Open follow-ups

- **Beat Transformer post-processing.** Our wrapper (`models/beat_transformer.py`) picks beats with
  SciPy `find_peaks` and does not import madmom. Upstream's reference inference instead uses madmom's
  `DBNBeatTrackingProcessor` / `DBNDownBeatTrackingProcessor`, and runs Spleeter (MIT) 5-stem demixing
  first; our wrapper feeds the same spectrogram to all 5 stems. If the Pro path adopts upstream's
  DBN decoding, it needs a madmom-free implementation (or a madmom install without its bundled models,
  after a license review). Do not re-add `madmom` to `requirements.txt`.
- The submodules were not checked out when this file was written. Re-confirm the weights' license
  files when bumping either submodule.
