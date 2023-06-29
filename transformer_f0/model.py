import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import os
import yaml
from torchaudio.transforms import Resample

from .pcmer import PCmer
from .nvSTFT import STFT


class TransformerF0(nn.Module):
    def __init__(
            self,
            input_channel=128,
            out_dims=1,
            n_layers=12,
            n_chans=512,
    ):
        super().__init__()

        # conv in stack
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, n_chans, 3, 1, 1),
            nn.GroupNorm(4, n_chans),
            nn.LeakyReLU(),
            nn.Conv1d(n_chans, n_chans, 3, 1, 1))

        # transformer
        self.decoder = PCmer(
            num_layers=n_layers,
            num_heads=8,
            dim_model=n_chans,
            dim_keys=n_chans,
            dim_values=n_chans,
            residual_dropout=0.1,
            attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)

        # out
        self.n_out = out_dims
        self.dense_out = weight_norm(
            nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)
        if not infer:
            x = 10 * F.mse_loss(x, (1 + gt_f0 / 700).log())
        return x


class TransformerF0Infer:
    def __init__(self, model_path, device=None):
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        with open(config_file, "r") as config:
            args = yaml.safe_load(config)
        self.args = DotDict(args)
        model = TransformerF0(
            input_channel=self.args.model.input_channel,
            out_dims=1,
            n_layers=self.args.model.n_layers,
            n_chans=self.args.model.n_chans,
        )
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        model.to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        self.model = model
        self.wav2mel = Wav2Mel(self.args)

    @torch.no_grad()
    def __call__(self, audio, sr):
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = self.wav2mel(audio=audio, sample_rate=sr)
        mel_f0 = self.model(mel=mel, infer=True)
        f0 = (mel_f0.exp() - 1) * 700
        return f0


class Wav2Mel:
    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.stft = STFT(
            self.args.mel.sampling_rate,
            self.args.mel.num_mels,
            self.args.mel.n_fft,
            self.args.mel.win_size,
            self.args.mel.hop_size,
            self.args.mel.fmin,
            self.args.mel.fmax
        )
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)  # B, n_frames, bins
        return mel

    def extract_mel(self, audio, sample_rate, keyshift=0):
        # resample
        if sample_rate == self.args.mel.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.args.mel.sampling_rate,
                                                         lowpass_filter_width=128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # extract
        mel = self.extract_nvstft(audio_res, keyshift=keyshift)  # B, n_frames, bins
        n_frames = int(audio.shape[1] // self.args.mel.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel

    def __call__(self, audio, sample_rate, keyshift=0):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift)


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
