import os
import random
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import noise_utils
from utils.os_utils import traverse_dir
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import pandas as pd


class TestDataset(Dataset):
    def __init__(
        self,
        path_root,
        extensions=["wav"],
        sample_rate=16000,
        wav2mel=None,
        whole_audio=False,
        load_all_data=True,
        #
        hop_size=160,
        #
        noise_ratio=0.7,
        brown_noise_ratio=1.0,
        load_data_num_processes=1,
        snb_noise=None,
        noise_beta=0,
        #
        device="cpu",
    ):
        super().__init__()
        self.wav2mel = wav2mel
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.noise_ratio = noise_ratio if noise_ratio is not None else 0.7
        self.brown_noise_ratio = (
            brown_noise_ratio if brown_noise_ratio is not None else 1.0
        )
        self.load_all_data = load_all_data
        self.snb_noise = snb_noise
        self.noise_beta = noise_beta

        self.paths = traverse_dir(
            os.path.join(path_root, "audio"),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True,
        )
        self.whole_audio = whole_audio
        self.data_buffer = {}
        self.device = device
        if load_all_data:
            print("Load all the data from :", path_root)
        else:
            print("Load the f0, volume data from :", path_root)

        with torch.no_grad():
            with ProcessPoolExecutor(max_workers=load_data_num_processes) as executor:
                tasks = []
                for i in range(load_data_num_processes):
                    start = int(i * len(self.paths) / load_data_num_processes)
                    end = int((i + 1) * len(self.paths) / load_data_num_processes)
                    file_chunk = self.paths[start:end]
                    tasks.append(file_chunk)
                for data_buffer in executor.map(self.load_data, tasks):
                    self.data_buffer.update(data_buffer)

            self.paths = np.array(self.paths, dtype=object)
            self.data_buffer = pd.DataFrame(self.data_buffer)

    def load_data(self, paths):
        with torch.no_grad():
            data_buffer = {}
            rank = mp.current_process()._identity
            rank = rank[0] if len(rank) > 0 else 0
            for name_ext in tqdm(paths):
                path_audio = os.path.join(self.path_root, "audio", name_ext)
                duration = librosa.get_duration(
                    filename=path_audio, sr=self.sample_rate
                )

                path_f0 = os.path.join(self.path_root, "f0", name_ext) + ".npy"
                f0 = np.load(path_f0)[:, None]

                if self.load_all_data:
                    """
                    audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio)
                    audio = torch.from_numpy(audio).to(device)
                    """
                    path_audio = (
                        os.path.join(self.path_root, "npaudiodir", name_ext) + ".npy"
                    )
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
        with torch.no_grad():
            name_ext = self.paths[file_idx]
            data_buffer = self.data_buffer[name_ext]

            return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        with torch.no_grad():
            name = os.path.splitext(name_ext)[0]
            frame_resolution = self.hop_size / self.sample_rate
            duration = data_buffer[0]
            waveform_sec = duration if self.whole_audio else self.waveform_sec

            idx_from = (
                0
                if self.whole_audio
                else random.uniform(0, duration - waveform_sec - 0.1)
            )
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(waveform_sec / frame_resolution)

            f0 = data_buffer[1].copy()
            f0 = torch.from_numpy(f0).float().cpu()

            if len(data_buffer) == 2:
                import librosa

                path_audio = os.path.join(self.path_root, "audio", name_ext)
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
            else:
                audio = data_buffer[2].copy()

            if self.snb_noise is not None:
                audio = noise_utils.add_noise_snb(
                    audio, self.snb_noise, self.noise_beta
                )

            peak = np.abs(audio).max()
            audio = 0.98 * audio / peak
            audio = torch.from_numpy(audio).float().unsqueeze(0).cpu()
            with torch.no_grad():
                mel = (
                    self.wav2mel(
                        audio,
                        sample_rate=self.sample_rate,
                        train=True,
                    )
                    .squeeze(0)
                    .cpu()
                )

            mel = mel[start_frame : start_frame + units_frame_len].detach()

            f0_frames = f0[start_frame : start_frame + units_frame_len].detach()

            del audio

            return mel, f0_frames, name, name_ext

    def __len__(self):
        return len(self.paths)
