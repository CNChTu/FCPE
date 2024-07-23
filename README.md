<h1 align="center">TorchFCPE</h1>

## Overview

TorchFCPE is a PyTorch-based library designed for audio pitch extraction and MIDI conversion. This README provides a quick guide on how to use the library for audio pitch inference and MIDI extraction.

## Installation

Before using the library, make sure you have the necessary dependencies installed:

```bash
pip install torchfcpe
```

## Usage

### 1. Audio Pitch Inference

```python
from torchfcpe import spawn_bundled_infer_model
import torch
import librosa

# Configure device and target hop size
device = 'cpu'  # or 'cuda' if using a GPU
sr = 16000  # Sample rate
hop_size = 160  # Hop size for processing

# Load and preprocess audio
audio, sr = librosa.load('test.wav', sr=sr)
audio = librosa.to_mono(audio)
audio_length = len(audio)
f0_target_length = (audio_length // hop_size) + 1
audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)

# Load the model
model = spawn_bundled_infer_model(device=device)

# Perform pitch inference
f0 = model.infer(
    audio,
    sr=sr,
    decoder_mode='local_argmax',  # Recommended mode
    threshold=0.006,  # Threshold for V/UV decision
    f0_min=80,  # Minimum pitch
    f0_max=880,  # Maximum pitch
    interp_uv=False,  # Interpolate unvoiced frames
    output_interp_target_length=f0_target_length,  # Interpolate to target length
)

print(f0)
```

### 2. MIDI Extraction

```python
# Extract MIDI from audio
midi = model.extact_midi(
    audio,
    sr=sr,
    decoder_mode='local_argmax',  # Recommended mode
    threshold=0.006,  # Threshold for V/UV decision
    f0_min=80,  # Minimum pitch
    f0_max=880,  # Maximum pitch
    output_path="test.mid",  # Save MIDI to file
)

print(midi)
```

### Notes

- **Inference Parameters:**

  - `audio`: Input audio as a `torch.Tensor`.
  - `sr`: Sample rate of the audio.
  - `decoder_mode` (Optional): Mode for decoding, 'local_argmax' is recommended.
  - `threshold` (Optional): Threshold for voice/unvoiced decision; default is 0.006.
  - `f0_min` (Optional): Minimum pitch value; default is 80 Hz.
  - `f0_max` (Optional): Maximum pitch value; default is 880 Hz.
  - `interp_uv` (Optional): Whether to interpolate unvoiced frames; default is False.
  - `output_interp_target_length` (Optional): Length to which the output pitch should be interpolated.

- **MIDI Extraction Parameters:**
  - `audio`: Input audio as a `torch.Tensor`.
  - `sr`: Sample rate of the audio.
  - `decoder_mode` (Optional): Mode for decoding; 'local_argmax' is recommended.
  - `threshold` (Optional): Threshold for voice/unvoiced decision; default is 0.006.
  - `f0_min` (Optional): Minimum pitch value; default is 80 Hz.
  - `f0_max` (Optional): Maximum pitch value; default is 880 Hz.
  - `output_path` (Optional): File path to save the MIDI file. If not provided, only returns the MIDI structure.
  - `tempo` (Optional): BPM for the MIDI file. If None, BPM is automatically predicted.

## Additional Features

- **Model as a PyTorch Module:**
  You can use the model as a standard PyTorch module. For example:

  ```python
  # Change device
  model = model.to(device)

  # Compile model
  model = torch.compile(model)
  ```
