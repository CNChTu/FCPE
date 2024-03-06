import torch
from .mel_extractor import Wav2Mel, Wav2MelModule
import pathlib


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


def spawn_wav2mel(args: DotDict, device: str = None) -> Wav2MelModule:
    """Spawn wav2mel"""
    _type = args.mel.type
    if (str(_type).lower() == 'none') or (str(_type).lower() == 'default'):
        _type = 'default'
    elif str(_type).lower() == 'stft':
        _type = 'stft'
    else:
        raise ValueError(f'  [ERROR] torchfcpe.tools.args.spawn_wav2mel: {_type} is not a supported args.mel.type')
    wav2mel = Wav2MelModule(
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
        clip_val=1e-05,
        mel_type=_type,
    )
    device = catch_none_args_opti(
        device,
        default='cpu',
        func_name='torchfcpe.tools.spawn_wav2mel',
        warning_str='.device is None',
    )
    return wav2mel.to(torch.device(device))


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
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        print(f'  [INFO]: Using {device} automatically.')
        print(f'  [INFO]    > call by: {func_name}')
    else:
        print(f'  [INFO]: device is not None, use {device}')
        print(f'  [INFO]    > call by:{func_name}')
        device = device

    # Check if the specified device is available, if not, switch to cpu
    if ((device == 'cuda' and not torch.cuda.is_available()) or
            (device == 'mps' and not torch.backends.mps.is_available())):
        print(f'  [WARN]: Specified device ({device}) is not available, switching to cpu.')
        device = 'cpu'

    return device


def get_config_json_in_same_path(path: str) -> str:
    """Get config json in same path"""
    path = pathlib.Path(path)
    config_json = path.parent / 'config.json'
    if config_json.exists():
        return str(config_json)
    else:
        raise FileNotFoundError(f'  [ERROR] {config_json} not found.')
