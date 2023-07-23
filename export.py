import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from fcpe.model import FCPE
from fcpe.model import DotDict
import os
import yaml
import argparse

class MelSpectrogram_ONNX(nn.Module):
    def __init__(
            self,
            n_mel_channels,
            sampling_rate,
            win_length,
            hop_length,
            n_fft=None,
            mel_fmin=0,
            mel_fmax=None,
            clamp=1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, center=True):
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=audio.device),
            center=center,
            return_complex=False
        )
        magnitude = torch.sqrt(torch.sum(fft ** 2, dim=-1))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class FCPEONNX(nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        with open(config_file, "r") as config:
            args = yaml.safe_load(config)
        self.args = DotDict(args)
        args = self.args
        model = FCPE(
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
        self.wav2mel = MelSpectrogram_ONNX(
            args.mel.num_mels, args.mel.sampling_rate, args.mel.win_size, args.mel.hop_size, args.mel.n_fft, args.mel.fmin, args.mel.fmax
        )
    
    def forward(self, audio):
        mel = self.wav2mel(audio).transpose(1, 2).to(self.device)
        mel_f0 = self.model(mel=mel, infer=True, return_hz_f0=True)
        # f0 = (mel_f0.exp() - 1) * 700
        f0 = mel_f0
        return f0.squeeze(0)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="model_300000.pt",
        type=str,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="MELPE.onnx",
        type=str,
        help="path to the output onnx file",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == "__main__":
    cmd = parse_args()
    model_path = cmd.model
    output_path = cmd.output
    model = FCPEONNX(model_path)
    waveform = torch.randn(1, 1919810)
    torch.onnx.export(
        model,
        (
            waveform
        ),
        output_path,
        input_names=[
            'waveform'
        ],
        output_names=[
            'f0'
        ],
        dynamic_axes={
            'waveform': {
                1: 'n_samples'
            },
            'f0': {
                1: 'n_frames'
            }
        },
        opset_version=17
    )
    