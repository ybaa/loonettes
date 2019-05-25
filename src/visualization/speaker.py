import numpy as np
from scipy.io import wavfile
import glob
from src.statics.positionEnum import Positions
import subprocess


class Speaker:

    def __init__(self):
        sounds = []
        path = '../data/raw/sounds/*.wav'

        filenames = glob.glob(path)
        filenames.sort()

        for file in filenames:
            rate, data = wavfile.read(file)
            sounds.append(data)

        self.sounds = sounds

    def say_recognition(self, recognition_iter, position):

        if recognition_iter is None:
            return None

        data_panned = self.linear_panning(self.sounds[recognition_iter], position)

        rate = 44100

        wavfile.write('output.wav', rate, data_panned)

        subprocess.call(["ffplay", "-nodisp", "-autoexit", 'output.wav'])


    def linear_panning(self, data, position):
        max_val = 32768
        left_channel = data[:, 0] / max_val
        right_channel = data[:, 1] / max_val

        if position == Positions.RIGHT.value:
            pan_value = 100
        elif position == Positions.LEFT.value:
            pan_value = -100
        else:
            pan_value = 0
        right_amp = pan_value / 200 + 0.5
        left_amp = 1 - right_amp

        sig = np.dstack((left_channel * left_amp, right_channel * right_amp))[0]

        return sig
