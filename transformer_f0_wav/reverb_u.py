import librosa
from scipy import signal
import numpy as np
import os
import random
# Function: 用librosa库读取音频
def read_audio(path):
    y, sr = librosa.load(path, sr=16000)
    return y

# Function: 给声音添加混响
def add_pyreverb(clean_speech, rir):
    
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]
    return reverb_speech

# function: peak 到0dB
def normalize_audio(audio):
    peak = np.abs(audio).max()
    return (audio / peak)*0.6

# 得到目录下的所有文件路径
def get_file_list(dir = "./simulated_rirs_16k/largeroom"):
    file_list = []
    '''
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    '''
    for room in os.listdir(dir):
        if room == 'rir_list':
            continue
        if room == 'room_info':
            continue
        _dir = dir + '/' + room
        for file in os.listdir(_dir):
            file_list.append(_dir+'/'+file)
    return file_list

rir_list = get_file_list()

def 加点混响(audio, rir_dir_list = rir_list):
    rir = read_audio(random.choice(rir_dir_list))
    rir = normalize_audio(rir)
    reverb_speech = add_pyreverb(audio, rir)
    reverb_speech = normalize_audio(reverb_speech)
    del rir
    return reverb_speech