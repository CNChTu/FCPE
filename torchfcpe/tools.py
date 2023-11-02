import torch
from .mel_extractor import Wav2Mel
from .models import CFNaiveMelPE


class DotDict(dict):
    """
    DotDict, used for config

    Example:
        # >>> config = DotDict({'a': 1, 'b': {'c': 2}}})
        # >>> config.a
        # 1
        # >>> config.b.c
        # 2
    """

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class InferCFNaiveMelPE:
    """Infer CFNaiveMelPE
    Args:
        args (DotDict): Config.
        state_dict (dict): Model state dict.
        device (str): Device. must be not None.
    """

    def __init__(self, args, state_dict, device):
        self.device = device
        self.wav2mel = spawn_wav2mel(args, device=self.device)
        self.model = spawn_model(args)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def __call__(self,
                 wav: torch.Tensor,
                 sr: [int, float],
                 decoder_mode: str = 'local_argmax',
                 threshold: float = 0.006
                 ) -> torch.Tensor:
        """Infer
        Args:
            wav (torch.Tensor): Input wav, (B, n_sample, 1).
            sr (int, float): Sample rate.
            decoder_mode (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
        return: f0 (torch.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1), 1).
        """
        with torch.no_grad():
            mel = self.wav2mel(wav, sr)
            f0 = self.model.infer(mel, decoder=decoder_mode, threshold=threshold)
        return f0  # (B, T, 1)


def spawn_infer_model_from_pt(pt_path: str, device: str = None) -> InferCFNaiveMelPE:
    """
    Spawn infer model from pt file
    Args:
        pt_path (str): Path to pt file.
        device (str): Device. Default: None.
    """
    device = get_device(device, 'torchfcpe.tools.spawn_infer_cf_naive_mel_pe_from_pt')
    ckpt = torch.load(pt_path, map_location=torch.device(device))
    args = DotDict(ckpt['config_dict'])
    if args.model.type == 'CFNaiveMelPE':
        infer_model = InferCFNaiveMelPE(args, ckpt['model'], device)
    else:
        raise ValueError(f'  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE')
    return infer_model


def spawn_model(args: DotDict) -> CFNaiveMelPE:
    """Spawn conformer naive model"""
    if args.model.type == 'CFNaiveMelPE':
        pe_model = CFNaiveMelPE(
            input_channels=catch_none_args_must(
                args.mel.num_mels,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.mel.num_mels is None',
            ),
            out_dims=catch_none_args_must(
                args.model.out_dims,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.out_dims is None',
            ),
            hidden_dims=catch_none_args_must(
                args.model.hidden_dims,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.hidden_dims is None',
            ),
            n_layers=catch_none_args_must(
                args.model.n_layers,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.n_layers is None',
            ),
            n_heads=catch_none_args_must(
                args.model.n_heads,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.n_heads is None',
            ),
            f0_max=catch_none_args_must(
                args.model.f0_max,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.f0_max is None',
            ),
            f0_min=catch_none_args_must(
                args.model.f0_min,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.f0_min is None',
            ),
            use_fa_norm=catch_none_args_must(
                args.model.use_fa_norm,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.use_fa_norm is None',
            ),
            residual_dropout=catch_none_args_must(
                args.model.residual_dropout,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.residual_dropout is None',
            ),
            attention_dropout=catch_none_args_must(
                args.model.attention_dropout,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.attention_dropout is None',
            ),
            conv_only=catch_none_args_opti(
                args.model.conv_only,
                default=False,
                func_name='torchfcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.conv_only is None',
            ),
        )
    else:
        raise ValueError(f'  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE')
    return pe_model


def spawn_wav2mel(args: DotDict, device: str = None) -> Wav2Mel:
    """Spawn wav2mel"""
    wav2mel = Wav2Mel(
        sr=catch_none_args_opti(
            args.mel.sr,
            default=16000,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.sr is None',
        ),
        n_mels=catch_none_args_opti(
            args.mel.num_mels,
            default=128,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.num_mels is None',
        ),
        n_fft=catch_none_args_opti(
            args.mel.n_fft,
            default=1024,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.n_fft is None',
        ),
        win_size=catch_none_args_opti(
            args.mel.win_size,
            default=1024,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.win_size is None',
        ),
        hop_length=catch_none_args_opti(
            args.mel.hop_size,
            default=160,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.hop_size is None',
        ),
        fmin=catch_none_args_opti(
            args.mel.fmin,
            default=0,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.fmin is None',
        ),
        fmax=catch_none_args_opti(
            args.mel.fmax,
            default=8000,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.fmax is None',
        ),
        clip_val=catch_none_args_opti(
            args.mel.clip_val,
            default=1e-5,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.clip_val is None',
        ),
        device=catch_none_args_opti(
            device,
            default='cpu',
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='.device is None',
        ),
    )
    return wav2mel


def catch_none_args_opti(x, default, func_name, warning_str=None, level='WARN'):
    """Catch None, optional"""
    if x is None:
        if warning_str is not None:
            print(f'  [{level}] {warning_str}; use default {default}')
            print(f'  [{level}]    > call by:{func_name}')
        return default
    else:
        return x


def catch_none_args_must(x, func_name, warning_str):
    """Catch None, must"""
    level = "ERROR"
    if x is None:
        print(f'  [{level}] {warning_str}')
        print(f'  [{level}]    > call by:{func_name}')
        raise ValueError(f'  [{level}] {warning_str}')
    else:
        return x


def get_device(device: str, func_name: str) -> str:
    """Get device"""
    if device is None:
        print(f'  [INFO] torchcrepe.tools.get_device: device is None, use auto choice.')
        if torch.cuda.is_available():
            device = 'cuda'
            print(f'  [INFO] torchcrepe.tools.get_device: cuda is available, use cuda.')
            print(f'  [INFO]    > call by:{func_name}')
        else:
            device = 'cpu'
            print(f'  [INFO] torchcrepe.tools.get_device: cuda is not available, use cpu.')
            print(f'  [INFO]    > call by:{func_name}')
    else:
        print(f'  [INFO] torchcrepe.tools.get_device: device is not None, use {device}')
        print(f'  [INFO]    > call by:{func_name}')
        device = device
    return device
