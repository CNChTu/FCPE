import os
import random
import colorednoise as cn
import re
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from transformer_f0_wav.model_with_bce import Wav2Mel
import transformer_f0_wav.utils as ut
import transformer_f0_wav.reverb_u as ru
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import pandas as pd

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
    wav2mel = Wav2Mel(args)
    data_train = F0Dataset(
        path_root=args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.mel.hop_size,
        sample_rate=args.mel.sampling_rate,
        duration=args.data.duration,
        load_all_data=args.train.cache_all_data,
        whole_audio=False,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        wav2mel=wav2mel,
        aug_noise=args.train.aug_noise,
        noise_ratio=args.train.noise_ratio,
        brown_noise_ratio=args.train.brown_noise_ratio,
        aug_flip=args.train.aug_flip,
        aug_mask=args.train.aug_mask,
        aug_mask_v_o=args.train.aug_mask_v_o,
        aug_mask_vertical_factor=args.train.aug_mask_vertical_factor,
        aug_mask_vertical_factor_v_o=args.train.aug_mask_vertical_factor_v_o,
        aug_mask_iszeropad_mode=args.train.aug_mask_iszeropad_mode,
        aug_mask_block_num=args.train.aug_mask_block_num,
        aug_mask_block_num_v_o=args.train.aug_mask_block_num_v_o,
        aug_eq=args.train.aug_eq,
        aug_reverb=args.train.aug_reverb,
        load_data_num_processes = 4
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
        duration=args.data.duration,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        wav2mel=wav2mel,
        aug_noise=args.train.aug_noise,
        noise_ratio=args.train.noise_ratio,
        brown_noise_ratio=args.train.brown_noise_ratio,
        aug_flip=args.train.aug_flip,
        aug_mask=args.train.aug_mask,
        aug_mask_v_o=args.train.aug_mask_v_o,
        aug_mask_vertical_factor=args.train.aug_mask_vertical_factor,
        aug_mask_vertical_factor_v_o=args.train.aug_mask_vertical_factor_v_o,
        aug_mask_iszeropad_mode=args.train.aug_mask_iszeropad_mode,
        aug_mask_block_num=args.train.aug_mask_block_num,
        aug_mask_block_num_v_o=args.train.aug_mask_block_num_v_o,
        aug_eq=args.train.aug_eq,
        aug_reverb=args.train.aug_reverb
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
            duration,
            load_all_data=True,
            whole_audio=False,
            extensions=['wav'],
            n_spk=1,
            device='cpu',
            wav2mel=None,
            aug_noise=False,
            noise_ratio=0.7,
            brown_noise_ratio=1.,
            aug_flip=False,
            aug_mask=False,
            aug_mask_v_o=False,
            aug_mask_vertical_factor=0.05,
            aug_mask_vertical_factor_v_o=0.3,
            aug_mask_iszeropad_mode='randon',  # randon zero or noise
            aug_mask_block_num=1,
            aug_mask_block_num_v_o=4,
            aug_keyshift=True,
            keyshift_min=-12,
            keyshift_max=12,
            aug_eq=True,
            aug_reverb=True,
            load_data_num_processes = 1
    ):
        super().__init__()
        self.wav2mel = wav2mel
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.duration = duration
        self.aug_noise = aug_noise if aug_noise is not None else False
        self.noise_ratio = noise_ratio if noise_ratio is not None else 0.7
        self.brown_noise_ratio = brown_noise_ratio if brown_noise_ratio is not None else 1.
        self.aug_flip = aug_flip if aug_flip is not None else False
        self.aug_mask = aug_mask if aug_mask is not None else False
        self.aug_mask_v_o = aug_mask_v_o if aug_mask_v_o is not None else False
        self.aug_mask_vertical_factor = aug_mask_vertical_factor if aug_mask_vertical_factor is not None else 0.05
        self.aug_mask_vertical_factor_v_o = aug_mask_vertical_factor_v_o if aug_mask_vertical_factor_v_o is not None else 0.3
        self.aug_mask_iszeropad_mode = aug_mask_iszeropad_mode if aug_mask_iszeropad_mode is not None else 'randon'
        self.aug_mask_block_num = aug_mask_block_num if aug_mask_block_num is not None else 1
        self.aug_mask_block_num_v_o = aug_mask_block_num_v_o if aug_mask_block_num_v_o is not None else 4
        self.aug_keyshift = aug_keyshift if aug_keyshift is not None else True
        self.keyshift_min = keyshift_min if keyshift_min is not None else -12
        self.keyshift_max = keyshift_max if keyshift_max is not None else 12
        self.aug_eq = aug_eq if aug_eq is not None else True
        self.aug_reverb = aug_reverb if aug_reverb is not None else True
        self.n_spk = n_spk
        self.device = device
        self.load_all_data = load_all_data

        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.data_buffer = {}
        self.device = device
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)

        with ProcessPoolExecutor(max_workers=load_data_num_processes) as executor:
            tasks = []
            for i in range(load_data_num_processes):
                start = int(i * len(self.paths) / load_data_num_processes)
                end = int((i + 1) * len(self.paths) / load_data_num_processes)
                file_chunk = self.paths[start:end]
                tasks.append(file_chunk)
            for data_buffer in executor.map(self.load_data, tasks):
                self.data_buffer.update(data_buffer)


        self.paths = np.array(self.paths, dtype = object)
        self.data_buffer = pd.DataFrame(self.data_buffer)

    def load_data(self, paths):
        data_buffer = {}
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        for name_ext in tqdm(paths):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            duration = librosa.get_duration(filename=path_audio, sr=self.sample_rate)

            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)[:, None]
            # f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(self.device)

            if self.n_spk is not None and self.n_spk > 1:
                dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                t_spk_id = spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                if spk_id < 1 or spk_id > self.n_spk:
                    raise ValueError(
                        ' [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk ')
            else:
                spk_id = 1
                t_spk_id = spk_id
            # spk_id = torch.LongTensor(np.array([spk_id])).to(self.device)
            spk_id = np.array([spk_id])

            if self.load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_audio = os.path.join(self.path_root, 'npaudiodir', name_ext) + '.npy'
                audio = np.load(path_audio)

                """
                data_buffer[name_ext] = {
                    'duration': duration,
                    'audio': audio,
                    'f0': f0,
                    'spk_id': spk_id,
                    't_spk_id': t_spk_id,
                }
                """
                data_buffer[name_ext] = (duration, f0, audio)
            else:
                """
                data_buffer[name_ext] = {
                    'duration': duration,
                    'f0': f0,
                    'spk_id': spk_id,
                    't_spk_id': t_spk_id
                }
                """
                data_buffer[name_ext] = (duration, f0)
        return data_buffer

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer[0] < (self.waveform_sec + 0.1):
            return self.__getitem__((file_idx + 1) % len(self.paths))

        # get item
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer[0]
        waveform_sec = duration if self.whole_audio else self.waveform_sec

        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)

        # load f0
        f0 = data_buffer[1]
        f0 = torch.from_numpy(f0).float().to(self.device)

        # load mel
        #audio = data_buffer.get('audio')
        if len(data_buffer) == 2:
            path_audio = os.path.join(self.path_root, 'npaudiodir', name_ext) + '.npy'
            audio = np.load(path_audio)
        else:
            audio = data_buffer[2]
        
        if random.choice((False, True)) and self.aug_eq:
            audio = ut.random_eq(audio, self.sample_rate)
            audio = ut.random_eq(audio, self.sample_rate)

        if random.choice((False, True)) and self.aug_eq:
            audio = ut.random_eq(audio, self.sample_rate)

        if random.choice((False, True)) and self.aug_eq:
            audio = ut.random_eq(audio, self.sample_rate)

        if random.random()<0.2 and self.aug_reverb:
            audio = ru.加点混响(audio)

        if random.choice((False, True)) and self.aug_keyshift:
            keyshift = random.uniform(self.keyshift_min, self.keyshift_max)
            f0 = 2 ** (keyshift / 12) * f0
        else:
            keyshift = 0
        
        is_aug_noise = bool(random.randint(0, 1))
        if self.aug_noise and is_aug_noise:
            if bool(random.randint(0, 1)):
                audio = ut.add_noise(audio, noise_ratio=self.noise_ratio)
            else:
                audio = ut.add_noise_slice(audio,self.sample_rate,self.duration,noise_ratio=self.noise_ratio, brown_noise_ratio=self.brown_noise_ratio)
                
        peak = np.abs(audio).max()
        audio = 0.98 * audio / peak
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = self.wav2mel(audio, sample_rate=self.sample_rate, keyshift=keyshift, train=True).squeeze(0)

        if self.aug_mask and bool(random.randint(0, 1)) and not is_aug_noise:
            v_o = bool(random.randint(0, 1)) and self.aug_mask_v_o
            mel = mel.transpose(-1,-2)
            if self.aug_mask_iszeropad_mode == 'zero':
                iszeropad = True
            elif self.aug_mask_iszeropad_mode =='noise':
                iszeropad = False
            else:
                iszeropad = bool(random.randint(0, 1))
            mel = ut.add_mel_mask_slice(mel,self.sample_rate,self.duration,hop_size=self.hop_size,vertical_factor=self.aug_mask_vertical_factor_v_o if v_o else self.aug_mask_vertical_factor,vertical_offset=v_o,iszeropad=iszeropad,block_num=self.aug_mask_block_num_v_o if v_o else self.aug_mask_block_num)
            mel = mel.transpose(-1,-2)
            
        mel = mel[start_frame: start_frame + units_frame_len]

        f0_frames = f0[start_frame: start_frame + units_frame_len]

        # load spk_id
        #spk_id = data_buffer.get('spk_id')
        #spk_id = torch.LongTensor(spk_id).to(self.device)

        if random.choice((False, False, True)) and self.aug_flip:
            f0_frames = torch.flip(f0_frames, dims=[0])
            mel = torch.flip(mel, dims=[0])
        
        del audio
        #return dict(mel=mel, f0=f0_frames, spk_id=spk_id, name=name, name_ext=name_ext)
        output = (mel, f0_frames, name, name_ext)
        return output

    def __len__(self):
        return len(self.paths)
