import colorednoise as cn
import random
import numpy as np
from sklearn.metrics import mean_squared_error

def add_noise(wav):
   beta = random.randint(0,2) # the exponent
   y = cn.powerlaw_psd_gaussian(beta, wav.shape[0])
   m =  np.sqrt(mean_squared_error(wav, np.zeros_like(y)))
   
   if beta == 0 or beta == 1:
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
      