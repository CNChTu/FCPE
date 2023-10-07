from .mel_extractor  import Wav2Mel


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


def spawn_wav2mel(args_mel: DotDict) -> Wav2Mel:
    """Spawn wav2mel instance"""
    wav2mel = Wav2Mel(
        sr=catch_none_args(
            args_mel.sr,
            16000,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: sr is None, use default 16000'),
        n_mels=catch_none_args(
            args_mel.n_mels,
            128,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: n_mels is None, use default 128'),
        n_fft=catch_none_args(
            args_mel.n_fft,
            1024,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: n_fft is None, use default 1024'),
        win_size=catch_none_args(
            args_mel.win_size,
            1024,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: win_size is None, use default 1024'),
        hop_length=catch_none_args(
            args_mel.hop_length,
            160,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: hop_length is None, use default 160'),
        fmin=catch_none_args(
            args_mel.fmin,
            0,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: fmin is None, use default 0'),
        fmax=catch_none_args(
            args_mel.fmax,
            8000,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: fmax is None, use default 8000'),
        clip_val=catch_none_args(
            args_mel.clip_val,
            1e-5,
            '  [WARN] torchcrepe.tools.spawn_wav2mel: clip_val is None, use default 1e-5'),
        device=catch_none_args(
            args_mel.device,
            'cpu',
            '  [WARN] torchcrepe.tools.spawn_wav2mel: device is None, use default \'cpu\'.')
    )
    return wav2mel


def catch_none_args(x, default, warning_str=None):
    """Catch None"""
    if x is None:
        if warning_str is not None:
            print(warning_str)
        return default
    else:
        return x
