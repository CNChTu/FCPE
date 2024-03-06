import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import Resample

import os

os.environ["LRU_CACHE_CAPACITY"] = "3"

try:
    from librosa.filters import mel as librosa_mel_fn
except ImportError:
    print('  [INF0] torchfcpe.mel_tools.nv_mel_extractor: Librosa not found,'
          ' use torchfcpe.mel_tools.mel_fn_librosa instead.')
    from .mel_fn_librosa import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class HannWindow(torch.nn.Module):
    def __init__(self, win_size):
        super().__init__()
        self.register_buffer('window', torch.hann_window(win_size), persistent=False)

    def forward(self):
        return self.window


class MelModule(torch.nn.Module):
    """Mel extractor

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): Minimum frequency. Defaults to 0.
        fmax (float, optional): Maximum frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
    """

    def __init__(self,
                 sr: [int, float],
                 n_mels: int,
                 n_fft: int,
                 win_size: int,
                 hop_length: int,
                 fmin: float = None,
                 fmax: float = None,
                 clip_val: float = 1e-5,
                 out_stft: bool = False,
                 ):
        super().__init__()
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        # self.mel_basis = {}
        self.register_buffer(
            'mel_basis',
            torch.tensor(librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).float(),
            persistent=False
        )
        self.hann_window = torch.nn.ModuleDict()
        self.out_stft = out_stft

    @torch.no_grad()
    def __call__(self,
                 y: torch.Tensor,  # (B, T, 1)
                 key_shift: [int, float] = 0,
                 speed: [int, float] = 1,
                 center: bool = False,
                 no_cache_window: bool = False
                 ) -> torch.Tensor:  # (B, T, n_mels)
        """Get mel spectrogram

        Args:
            y (torch.Tensor): Input waveform, shape=(B, T, 1).
            key_shift (int, optional): Key shift. Defaults to 0.
            speed (int, optional): Variable speed enhancement factor. Defaults to 1.
            center (bool, optional): center for torch.stft. Defaults to False.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        clip_val = self.clip_val

        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        y = y.squeeze(-1)

        if torch.min(y) < -1.:
            print('[error with torchfcpe.mel_extractor.MelModule]min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('[error with torchfcpe.mel_extractor.MelModule]max value is ', torch.max(y))

        key_shift_key = str(key_shift)
        if not no_cache_window:
            if key_shift_key in self.hann_window:
                hann_window = self.hann_window[key_shift_key]
            else:
                hann_window = HannWindow(win_size_new).to(self.mel_basis.device)
                self.hann_window[key_shift_key] = hann_window
            hann_window_tensor = hann_window()
        else:
            hann_window_tensor = torch.hann_window(win_size_new).to(self.mel_basis.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)

        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new,
                          window=hann_window_tensor,
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        if key_shift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new
        if self.out_stft:
            spec = spec[:, :512, :]
        else:
            spec = torch.matmul(self.mel_basis, spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        spec = spec.transpose(-1, -2)
        return spec  # (B, T, n_mels)


class Wav2MelModule(torch.nn.Module):
    """
    Wav to mel converter
    NOTE: This class of code is reserved for training only, please use Wav2MelModule for inference

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): Minimum frequency. Defaults to 0.
        fmax (float, optional): Maximum frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
        device (str, optional): Device. Defaults to 'cpu'.
    """

    def __init__(self,
                 sr: [int, float],
                 n_mels: int,
                 n_fft: int,
                 win_size: int,
                 hop_length: int,
                 fmin: float = None,
                 fmax: float = None,
                 clip_val: float = 1e-5,
                 mel_type="default",
                 ):
        super().__init__()
        # catch None
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        # init
        self.sampling_rate = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        # self.device = device
        self.register_buffer(
            'tensor_device_marker',
            torch.tensor(1.0).float(),
            persistent=False
        )
        self.resample_kernel = torch.nn.ModuleDict()
        if mel_type == "default":
            self.mel_extractor = MelModule(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val,
                                           out_stft=False)
        elif mel_type == "stft":
            self.mel_extractor = MelModule(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val,
                                           out_stft=True)
        self.mel_type = mel_type

    def device(self):
        """Get device"""
        return self.tensor_device_marker.device

    @torch.no_grad()
    def __call__(self,
                 audio: torch.Tensor,  # (B, T, 1)
                 sample_rate: [int, float],
                 keyshift: [int, float] = 0,
                 no_cache_window: bool = False
                 ) -> torch.Tensor:  # (B, T, n_mels)
        """
        Get mel spectrogram

        Args:
            audio (torch.Tensor): Input waveform, shape=(B, T, 1).
            sample_rate (int): Sampling rate.
            keyshift (int, optional): Key shift. Defaults to 0.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        # resample
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                if len(self.resample_kernel) > 8:
                    self.resample_kernel.clear()
                self.resample_kernel[key_str] = Resample(
                    sample_rate,
                    self.sampling_rate,
                    lowpass_filter_width=128
                ).to(self.tensor_device_marker.device)
            audio_res = self.resample_kernel[key_str](audio.squeeze(-1)).unsqueeze(-1)

        # extract
        mel = self.mel_extractor(audio_res, keyshift, no_cache_window=no_cache_window)
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]

        return mel  # (B, T, n_mels)


class MelExtractor:
    """Mel extractor
    NOTE: This class of code is reserved for training only, please use MelModule for inference

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): Minimum frequency. Defaults to 0.
        fmax (float, optional): Maximum frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
    """

    def __init__(self,
                 sr: [int, float],
                 n_mels: int,
                 n_fft: int,
                 win_size: int,
                 hop_length: int,
                 fmin: float = None,
                 fmax: float = None,
                 clip_val: float = 1e-5,
                 out_stft: bool = False,
                 ):
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}
        self.out_stft = out_stft

    @torch.no_grad()
    def __call__(self,
                 y: torch.Tensor,  # (B, T, 1)
                 key_shift: [int, float] = 0,
                 speed: [int, float] = 1,
                 center: bool = False,
                 no_cache_window: bool = False
                 ) -> torch.Tensor:  # (B, T, n_mels)
        """Get mel spectrogram

        Args:
            y (torch.Tensor): Input waveform, shape=(B, T, 1).
            key_shift (int, optional): Key shift. Defaults to 0.
            speed (int, optional): Variable speed enhancement factor. Defaults to 1.
            center (bool, optional): center for torch.stft. Defaults to False.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        if not no_cache_window:
            mel_basis = self.mel_basis
            hann_window = self.hann_window
        else:
            mel_basis = {}
            hann_window = {}

        y = y.squeeze(-1)

        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        mel_basis_key = str(fmax) + '_' + str(y.device)
        if (mel_basis_key not in mel_basis) and (not self.out_stft):
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        key_shift_key = str(key_shift) + '_' + str(y.device)
        if key_shift_key not in hann_window:
            hann_window[key_shift_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)

        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new,
                          window=hann_window[key_shift_key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        if key_shift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new
        if self.out_stft:
            spec = spec[:, :512, :]
        else:
            spec = torch.matmul(mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        spec = spec.transpose(-1, -2)
        return spec  # (B, T, n_mels)


# init nv_mel_extractor cache
# will remove this when we have a better solution
# mel_extractor = MelExtractor(16000, 128, 1024, 1024, 160, 0, 8000)


class Wav2Mel:
    """
    Wav to mel converter
    NOTE: This class of code is reserved for training only, please use Wav2MelModule for inference

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): Minimum frequency. Defaults to 0.
        fmax (float, optional): Maximum frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
        device (str, optional): Device. Defaults to 'cpu'.
    """

    def __init__(self,
                 sr: [int, float],
                 n_mels: int,
                 n_fft: int,
                 win_size: int,
                 hop_length: int,
                 fmin: float = None,
                 fmax: float = None,
                 clip_val: float = 1e-5,
                 device='cpu',
                 mel_type="default",
                 ):
        # catch None
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        # init
        self.sampling_rate = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.device = device
        self.resample_kernel = {}
        if mel_type == "default":
            self.mel_extractor = MelExtractor(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val,
                                              out_stft=False)
        elif mel_type == "stft":
            self.mel_extractor = MelExtractor(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val,
                                              out_stft=True)
        self.mel_type = mel_type

    def device(self):
        """Get device"""
        return self.device

    @torch.no_grad()
    def __call__(self,
                 audio: torch.Tensor,  # (B, T, 1)
                 sample_rate: [int, float],
                 keyshift: [int, float] = 0,
                 no_cache_window: bool = False
                 ) -> torch.Tensor:  # (B, T, n_mels)
        """
        Get mel spectrogram

        Args:
            audio (torch.Tensor): Input waveform, shape=(B, T, 1).
            sample_rate (int): Sampling rate.
            keyshift (int, optional): Key shift. Defaults to 0.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        # resample
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate,
                    self.sampling_rate,
                    lowpass_filter_width=128
                ).to(self.device)
            audio_res = self.resample_kernel[key_str](audio.squeeze(-1)).unsqueeze(-1)

        # extract
        mel = self.mel_extractor(audio_res, keyshift, no_cache_window=no_cache_window)
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]

        return mel  # (B, T, n_mels)


def unit_text():
    """
    Test unit for nv_mel_extractor.py
    Should be set path to your test audio file.
    Need matplotlib and librosa to plot.
    require: pip install matplotlib librosa
    """
    import time

    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError:
        print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Matplotlib or Librosa not found,'
              ' skip plotting.')
        exit(1)

    # spawn mel extractor and wav2mel
    mel_extractor_test = MelExtractor(16000, 128, 1024, 1024, 160, 0, 8000)
    wav2mel_test = Wav2Mel(16000, 128, 1024, 1024, 160, 0, 8000)

    # load audio
    audio_path = r'E:\AUFSe04BPyProgram\AUFSd04BPyProgram\ddsp-svc\20230308\diffusion-svc\samples\GJ2.wav'
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(-1)
    audio = audio.to('cuda')
    print('  [UNIT_TEST] torchfcpe.mel_tools.mel_extractor: Audio shape: {}'.format(audio.shape))

    # test mel extractor
    start_time = time.time()
    mel1 = mel_extractor_test(audio, 0, 1, False)
    print('  [UNIT_TEST] torchfcpe.mel_extractor: Mel extractor time cost: {:.3f}s'.format(
        time.time() - start_time))
    print('  [UNIT_TEST] torchfcpe.mel_extractor: Mel extractor output shape: {}'.format(mel1.shape))

    # test wav2mel
    start_time = time.time()
    mel2 = wav2mel_test(audio, 16000, 0)
    print('  [UNIT_TEST] torchfcpe.mel_extractor: Wav2mel time cost: {:.3f}s'.format(
        time.time() - start_time))
    print('  [UNIT_TEST] torchfcpe.mel_extractor: Wav2mel output shape: {}'.format(mel2.shape))

    # test melModule
    mel_module = MelModule(16000, 128, 1024, 1024, 160, 0, 8000).to('cuda')
    mel3 = mel_module(audio, 0, 1, False).to('cuda')
    print('  [UNIT_TEST] torchfcpe.mel_extractor: MelModule output shape: {}'.format(mel3.shape))

    # test Wav2MelModule
    wav2mel_module = Wav2MelModule(16000, 128, 1024, 1024, 160, 0, 8000).to('cuda')
    mel4 = wav2mel_module(audio, 16000, 0).to('cuda')
    print('  [UNIT_TEST] torchfcpe.mel_extractor: Wav2MelModule output shape: {}'.format(mel4.shape))

    # plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 5, 1)
    librosa.display.waveshow(audio.squeeze().cpu().numpy(), sr=16000)
    plt.title('Audio')
    plt.subplot(1, 5, 2)
    librosa.display.specshow(mel1.squeeze().cpu().numpy().T, sr=16000, hop_length=160, x_axis='time', y_axis='mel')
    plt.title('Mel extractor')
    plt.subplot(1, 5, 3)
    librosa.display.specshow(mel2.squeeze().cpu().numpy().T, sr=16000, hop_length=160, x_axis='time', y_axis='mel')
    plt.title('Wav2mel')

    plt.subplot(1, 5, 4)
    librosa.display.specshow(mel3.squeeze().cpu().numpy().T, sr=16000, hop_length=160, x_axis='time', y_axis='mel')
    plt.title('MelModule')
    plt.subplot(1, 5, 5)
    librosa.display.specshow(mel4.squeeze().cpu().numpy().T, sr=16000, hop_length=160, x_axis='time', y_axis='mel')
    plt.title('Wav2MelModule')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    unit_text()
