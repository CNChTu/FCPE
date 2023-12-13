<h1 align="center">TorchFCPE</h1>
<div align="center">

</div>


## Useage

```python
from torchfcpe import spawn_bundled_infer_model
import torch
import librosa

# device
device = 'cpu'

# load audio
audio, sr = librosa.load('test.wav', sr=16000)
audio = librosa.to_mono(audio)
audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)

# load model
model = spawn_bundled_infer_model(device=device)

# infer
f0 = model.infer(
    audio,
    sr=sr,
    decoder_mode='local_argmax',
    threshold=0.006,
    f0_min=80,
    f0_max=880,
    interp_uv=False
)

print(f0)
```

