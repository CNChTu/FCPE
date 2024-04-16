<h1 align="center">TorchFCPE</h1>
<div align="center">

</div>


## Useage

```python
from torchfcpe import spawn_bundled_infer_model
import torch
import librosa

# configure device and target hop_size
device = 'cpu'
sr = 16000
hop_size = 160

# load audio
audio, sr = librosa.load('test.wav', sr=sr)
audio = librosa.to_mono(audio)
audio_length = len(audio)
f0_target_length=(audio_length // hop_size) + 1
audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)

# load model
model = spawn_bundled_infer_model(device=device)

# infer
'''
audio: wav, torch.Tensor
sr: sample rate
decoder_mode: [Optional] 'local_argmax' is recommended
threshold: [Optional] threshold for V/UV decision, 0.006 is recommended
f0_min: [Optional] minimum f0
f0_max: [Optional] maximum f0
interp_uv: [Optional] whether to interpolate unvoiced frames
output_interp_target_length: [Optional] If not None, the output f0 will be
    interpolated to the target length
'''
f0 = model.infer(
    audio,
    sr=sr,
    decoder_mode='local_argmax',
    threshold=0.006,
    f0_min=80,
    f0_max=880,
    interp_uv=False,
    output_interp_target_length=f0_target_length,
)

print(f0)

''' MIDI Extract '''
'''
audio: wav, torch.Tensor
sr: sample rate
decoder_mode: [Optional] 'local_argmax' is recommended
threshold: [Optional] threshold for V/UV decision, 0.006 is recommended
f0_min: [Optional] minimum f0
f0_max: [Optional] maximum f0
output_path: [Optional] If input, save midi; otherwise, only return midi structure
tempo: [Optional] BPM, if None, Automatic prediction of bpm
'''

midi = model.extact_midi(
    audio,
    sr=sr,
    decoder_mode='local_argmax',
    threshold=0.006,
    f0_min=80,
    f0_max=880,
    output_path="test.mid"
)

print(midi)


# the model is son class of torch.nn.Module, so you can use it as a normal pytorch model
# example: change device
model = model.to(device)
# example: compile model
model = torch.compile(model)

```

