import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import os
import yaml
from torchaudio.transforms import Resample
from siren import Sine
import numpy as np

from transformer_f0_wav.pcmer import PCmer
from transformer_f0_wav.pcmer_siren import PCmer as PCmerS
from transformer_f0_wav.pcmer_full import PCmer as PCmerF
from transformer_f0_wav.pcmer_siren_full import PCmer as PCmerFS
from transformer_f0_wav.nvSTFT import STFT


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class TransformerF0BCE(nn.Module):
    def __init__(
            self,
            input_channel=128,
            out_dims=360,
            n_layers=12,
            n_chans=512,
            use_siren=False,
            use_full=False,
            loss_mse_scale=10,
            loss_l2_regularization=False,
            loss_l2_regularization_scale=1,
            loss_grad1_mse=False,
            loss_grad1_mse_scale=1,
            f0_max=1975.5,
            f0_min=32.70,
            confidence=False,
            threshold=0.05,
            use_input_conv=True,
            residual_dropout=0.1,
            attention_dropout=0.1
    ):
        super().__init__()
        use_siren = False if (use_siren is None) else use_siren
        use_full = False if (use_full is None) else use_full
        self.loss_mse_scale = loss_mse_scale if (loss_mse_scale is not None) else 10
        self.loss_l2_regularization = loss_l2_regularization if (loss_l2_regularization is not None) else False
        self.loss_l2_regularization_scale = loss_l2_regularization_scale if (loss_l2_regularization_scale
                                                                             is not None) else 1
        self.loss_grad1_mse = loss_grad1_mse if (loss_grad1_mse is not None) else False
        self.loss_grad1_mse_scale = loss_grad1_mse_scale if (loss_grad1_mse_scale is not None) else 1
        self.f0_max = f0_max if (f0_max is not None) else 1975.5
        self.f0_min = f0_min if (f0_min is not None) else 32.70
        self.confidence = confidence if (confidence is not None) else False
        self.threshold = threshold if (threshold is not None) else 0.05
        self.use_input_conv = use_input_conv if (use_input_conv is not None) else True

        self.residual_dropout = residual_dropout if (residual_dropout is not None) else 0.1
        self.attention_dropout = attention_dropout if (attention_dropout is not None) else 0.1

        self.cent_table_b = torch.Tensor(
            np.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0],
                        out_dims))
        self.register_buffer("cent_table", self.cent_table_b)

        # conv in stack
        _leaky = Sine(w0=1) if use_siren else nn.LeakyReLU()
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, n_chans, 3, 1, 1),
            nn.GroupNorm(4, n_chans),
            _leaky,
            nn.Conv1d(n_chans, n_chans, 3, 1, 1))

        # transformer
        if use_siren:
            if use_full:
                self.decoder = PCmerFS(
                    num_layers=n_layers,
                    num_heads=8,
                    dim_model=n_chans,
                    dim_keys=n_chans,
                    dim_values=n_chans,
                    residual_dropout=self.residual_dropout,
                    attention_dropout=self.attention_dropout)
            else:
                self.decoder = PCmerS(
                    num_layers=n_layers,
                    num_heads=8,
                    dim_model=n_chans,
                    dim_keys=n_chans,
                    dim_values=n_chans,
                    residual_dropout=self.residual_dropout,
                    attention_dropout=self.attention_dropout)
        else:
            if use_full:
                self.decoder = PCmerF(
                    num_layers=n_layers,
                    num_heads=8,
                    dim_model=n_chans,
                    dim_keys=n_chans,
                    dim_values=n_chans,
                    residual_dropout=self.residual_dropout,
                    attention_dropout=self.attention_dropout)
            else:
                self.decoder = PCmer(
                    num_layers=n_layers,
                    num_heads=8,
                    dim_model=n_chans,
                    dim_keys=n_chans,
                    dim_values=n_chans,
                    residual_dropout=self.residual_dropout,
                    attention_dropout=self.attention_dropout)
        self.norm = nn.LayerNorm(n_chans)

        # out
        self.n_out = out_dims
        self.dense_out = weight_norm(
            nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        if self.use_input_conv:
            x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        else:
            x = mel
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)  # [B,N,D]
        x = torch.sigmoid(x)
        if not infer:
            gt_cent_f0 = self.f0_to_cent(gt_f0)  # mel f0  #[B,N,1]
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)  # #[B,N,out_dim]
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, gt_cent_f0)  # bce loss
            # l2 regularization
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all
        if infer:
            x = self.cents_decoder(x)
            x = self.cent_to_f0(x)
            if not return_hz_f0:
                x = (1 + x / 700).log()
        return x

    def cents_decoder(self, y ,mask = True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)  # cents: [B,N,1]
        if mask:
            confident = torch.max(y, dim=-1 ,keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn

    def cent_to_f0(self,cent):
        return 10. * 2 ** (cent / 1200.)
    
    def f0_to_cent(self,f0):
        return 1200. * torch.log2(f0/10.)

    def gaussian_blurred_cent(self, cents):  # cents: [B,N,1]
        mask = (cents>0.1) & (cents<(1200.*np.log2(self.f0_max/10.)))
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()


class TransformerF0Infer:
    def __init__(self, model_path, device=None):
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        with open(config_file, "r") as config:
            args = yaml.safe_load(config)
        self.args = DotDict(args)
        model = TransformerF0BCE(
            input_channel=self.args.model.input_channel,
            out_dims=self.args.model.out_dims,
            n_layers=self.args.model.n_layers,
            n_chans=self.args.model.n_chans,
            use_siren=self.args.model.use_siren,
            use_full=self.args.model.use_full,
            loss_mse_scale=self.args.loss.loss_mse_scale,
            loss_l2_regularization=self.args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=self.args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale,
            f0_max=self.args.model.f0_max,
            f0_min=self.args.model.f0_min,
            confidence=self.args.model.confidence,
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
        mel_f0 = self.model(mel=mel, infer=True, return_hz_f0=True)
        #f0 = (mel_f0.exp() - 1) * 700
        f0 = mel_f0
        return f0


class Wav2Mel:
    def __init__(self, args, device=None):
        #self.args = args
        self.sampling_rate = args.mel.sampling_rate
        self.hop_size = args.mel.hop_size
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.stft = STFT(
            args.mel.sampling_rate,
            args.mel.num_mels,
            args.mel.n_fft,
            args.mel.win_size,
            args.mel.hop_size,
            args.mel.fmin,
            args.mel.fmax
        )
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        mel = self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)  # B, n_frames, bins
        return mel

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        # resample
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.sampling_rate,
                                                         lowpass_filter_width=128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # extract
        mel = self.extract_nvstft(audio_res, keyshift=keyshift, train=train)  # B, n_frames, bins
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
