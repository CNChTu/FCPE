import os
import random
import re
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset


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


def get_data_loaders(args):
    data_train = F0Dataset(
        path_root=args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.mel.hop_size,
        sample_rate=args.mel.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=False,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device == 'cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device == 'cpu' else False,
        pin_memory=True if args.train.cache_device == 'cpu' else False
    )
    data_valid = F0Dataset(
        path_root=args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.mel.hop_size,
        sample_rate=args.mel.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid


class F0Dataset(Dataset):
    def __init__(
            self,
            path_root,
            waveform_sec,
            hop_size,
            sample_rate,
            load_all_data=True,
            whole_audio=False,
            extensions=['wav'],
            n_spk=1,
            device='cpu',
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.data_buffer = {}
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            name = os.path.splitext(name_ext)[0]
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            duration = librosa.get_duration(filename=path_audio, sr=self.sample_rate)

            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)

            if n_spk is not None and n_spk > 1:
                dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                t_spk_id = spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                if spk_id < 1 or spk_id > n_spk:
                    raise ValueError(
                        ' [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk ')
            else:
                spk_id = 1
                t_spk_id = spk_id
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            if load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)

                self.data_buffer[name_ext] = {
                    'duration': duration,
                    'mel': mel,
                    'f0': f0,
                    'spk_id': spk_id,
                    't_spk_id': t_spk_id,
                }
            else:
                self.data_buffer[name_ext] = {
                    'duration': duration,
                    'f0': f0,
                    'spk_id': spk_id,
                    't_spk_id': t_spk_id
                }

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__((file_idx + 1) % len(self.paths))

        # get item
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec

        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)

        # load mel
        mel = data_buffer.get('mel')
        if mel is None:
            mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
            mel = np.load(mel)
            mel = mel[start_frame: start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float()
        else:
            mel = mel[start_frame: start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        f0_frames = f0[start_frame: start_frame + units_frame_len]

        # load spk_id
        spk_id = data_buffer.get('spk_id')

        return dict(mel=mel, f0=f0_frames, spk_id=spk_id, name=name, name_ext=name_ext)

    def __len__(self):
        return len(self.paths)
