import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from tqdm import tqdm
from transformer_f0.model import Wav2Mel
import yaml
from transformer_f0.f0_others import F0_Extractor
from concurrent.futures import ProcessPoolExecutor
import transformer_f0_wav.utils as ut
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    return parser.parse_args(args=args, namespace=namespace)


    # run  
def process(file,path_srcdir,path_f0dir,path_npaudiodir,path_skipdir,path_worlddir,read_sr,uv_interp,f0_extractor,hop_size):
    binfile = file + '.npy'
    path_srcfile = os.path.join(path_srcdir, file)
    path_f0file = os.path.join(path_f0dir, binfile)
    path_npaudiofile = os.path.join(path_npaudiodir, binfile)
    path_skipfile = os.path.join(path_skipdir, file)

    # load audio
    audio, temp_sr = librosa.load(path_srcfile, sr=read_sr)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    # extract mel
    #mel = wav2mel(audio=torch.from_numpy(audio).float().unsqueeze(0).to(device), sample_rate=temp_sr).squeeze().to('cpu').numpy()

    # extract f0
    f0 = f0_extractor.extract(audio, uv_interp=False, sr=temp_sr)

    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        if uv_interp:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

        audio = ut.worldSynthesize(audio,temp_sr,hop_size,f0_in = f0)
    
        # save npy
        os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
        np.save(path_f0file, f0)
        os.makedirs(os.path.dirname(path_npaudiofile), exist_ok=True)
        np.save(path_npaudiofile, audio)
    else:
        print('\n[Error] F0 extraction failed: ' + path_srcfile)
        os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
        shutil.move(path_srcfile, os.path.dirname(path_skipfile))
        print('This file has been moved to ' + path_skipfile)

def loop_process(filelist, path_srcdir,path_f0dir,path_npaudiodir,path_skipdir,path_worlddir,read_sr,uv_interp,f0_extractor,hop_size=160):
    for file in tqdm(filelist, total=len(filelist)):
        process(file,path_srcdir,path_f0dir,path_npaudiodir,path_skipdir,path_worlddir,read_sr,uv_interp,f0_extractor,hop_size)

def preprocess(path, f0_extractor, wav2mel, uv_interp=False, read_sr=16000, device='cuda', extensions=['wav'], num_workers = 4,hop_size=160):
    path_srcdir = os.path.join(path, 'audio')
    path_f0dir = os.path.join(path, 'f0')
    path_npaudiodir = os.path.join(path, 'npaudiodir')
    path_skipdir = os.path.join(path, 'skip')
    path_worlddir = os.path.join(path, 'world')

    # list files
    filelist = traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)

    print('Preprocess the audio clips in :', path_srcdir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
       tasks = []
       for i in range(num_workers):
           start = int(i * len(filelist) / num_workers)
           end = int((i + 1) * len(filelist) / num_workers)
           file_chunk = filelist[start:end]
           tasks.append(executor.submit(loop_process, file_chunk,path_srcdir,path_f0dir,path_npaudiodir,path_skipdir,path_worlddir,read_sr,uv_interp,f0_extractor,hop_size))

       for task in tqdm(tasks):
           task.result()

    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load config
    with open(cmd.config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    extensions = args.data.extensions

    num_workers = 6

    # initialize f0 extractor
    f0_extractor = F0_Extractor(
        f0_extractor=args.data.f0_extractor,
        sample_rate=args.mel.sampling_rate,
        hop_size=args.mel.hop_size,
        f0_min=args.data.f0_min,
        f0_max=args.data.f0_max,
        block_size=args.mel.hop_size,
        model_sampling_rate=args.mel.sampling_rate
    )

    # initialize mel extractor
    #wav2mel = Wav2Mel(args)

    # preprocess training set
    preprocess(args.data.train_path, f0_extractor, None, uv_interp=args.data.us_uv, read_sr=args.mel.sampling_rate, device=device, extensions=['wav'], num_workers = num_workers,hop_size=args.mel.hop_size)

    # preprocess validation set
    preprocess(args.data.valid_path, f0_extractor, None, uv_interp=args.data.us_uv, read_sr=args.mel.sampling_rate, device=device, extensions=['wav'], num_workers = num_workers,hop_size=args.mel.hop_size)
