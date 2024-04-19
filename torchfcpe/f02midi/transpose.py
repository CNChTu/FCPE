# -*- coding: utf-8 -*-
# %%
import argparse
import numpy as np
from pathlib import Path
from .featureExtraction import *
from .quantization import *
from .utils import *
from .MIDI import *
import librosa

def f0_to_note(f0):
    """ convert frame-level pitch score(hz) to note-level (time-axis) """
    note = 69 + 12 * np.log2(f0 / 440 + 1e-4)
    note = np.round(note)
    note = note.astype(int)
    note[note < 0] = 0
    note[note > 127] = 127
    return note

def f02midi(f0, tempo = None, y = None, sr = None, output_path = None):
    """ f0 shape: (n_frames,) """

    if tempo is None:
        if y is not None:
            target_sr = 22050
            y = librosa.resample(y = y, orig_sr = sr, target_sr = target_sr)
            onset_strength = librosa.onset.onset_strength(y = y, sr=target_sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_strength, sr=target_sr)
        else:
            tempo = 120

    f0 = f0_to_note(f0)
    refined_fl_note = refine_note(f0, tempo)  # frame-level pitch score

    """ convert frame-level pitch score to note-level (time-axis) """
    segment = note_to_segment(refined_fl_note)  # note-level pitch score
    if output_path is None:
        return segment
    else:
        """ save ouput to .mid """
        segment_to_midi(segment, path_output=output_path, tempo=tempo)
        return segment
