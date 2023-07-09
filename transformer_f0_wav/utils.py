import colorednoise as cn
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import torch

def add_noise(wav):
   beta = random.random()*2 # the exponent
   y = cn.powerlaw_psd_gaussian(beta, wav.shape[0])
   m =  np.sqrt(mean_squared_error(wav, np.zeros_like(y)))
   
   if beta >= 0 and beta <= 1.5:
      wav += (0.7*random.random())* m*y
   else:
      wav += random.random()* m*y
   return wav

def add_noise_slice(wav, sr, duration, add_factor = 0.50):
   slice_length = int(duration * sr)
   n_frames = int(wav.shape[-1]//slice_length)
   slice_length_noise = int(slice_length * add_factor)
   for n in range(n_frames):
      left, right = int(n * slice_length), int((n + 1) * slice_length)
      offset = random.randint(left,right-slice_length_noise)
      if wav[offset:offset+slice_length_noise].shape[0] != 0:
         wav[offset:offset+slice_length_noise] = add_noise(wav[offset:offset+slice_length_noise])
   return wav
      
def add_mel_mask(mel, iszeropad= False, esp = 1e-5):
   if iszeropad:
      return torch.ones_like(mel)*esp
   else:
      return (random.random()*0.9+0.1)*torch.randn_like(mel)
   
def add_mel_mask_slice(mel, sr, duration,hop_size=512 ,add_factor = 0.3, vertical_offset = True, vertical_factor= 0.05, iszeropad = True, islog = True, block_num = 5 , esp = 1e-5):
   if islog:
      mel = torch.exp(mel)
   slice_length = int(duration * sr)//hop_size
   n_frames = int(mel.shape[-1]//slice_length)
   n_mels = mel.shape[-2]
   for n in range(n_frames):
      line_num = n_mels//block_num
      for i in range(block_num):
         now_vertical_factor = vertical_factor + random.random()*0.1
         now_add_factor = add_factor + random.random()*0.1
         slice_length_noise = int(slice_length * now_add_factor)
         if vertical_offset:
            v_offset = int(random.uniform(line_num * i, line_num*(i+1)-now_vertical_factor))
            n_v_down = v_offset
            n_v_up = int(v_offset+now_vertical_factor*n_mels)
         else:
            n_v_down = 0
            n_v_up = n_mels
         left, right = int(n * slice_length), int((n + 1) * slice_length)
         offset = int(random.uniform(left,right-slice_length_noise))
         if mel[n_v_down:n_v_up,offset:offset+slice_length_noise].shape[-1] != 0:
            mel[n_v_down:n_v_up,offset:offset+slice_length_noise] = add_mel_mask(mel[n_v_down:n_v_up,offset:offset+slice_length_noise], iszeropad, esp)
   if islog:
      mel = torch.log(torch.clamp(mel, min=esp))
   return mel