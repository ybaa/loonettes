import numpy as np
from scipy.io import wavfile
import glob
from src.statics.positionEnum import PositionsH, PositionV
import subprocess


class Speaker:

    def __init__(self):
        sounds = []
        path = '../data/raw/sounds/*.wav'

        filenames = glob.glob(path)
        filenames.sort()

        for file in filenames:
            self.rate, data = wavfile.read(file)
            sounds.append(data)

        self.sounds = sounds

    def say_recognition(self, recognition_iter, position_horizontal, position_vertical):

        if recognition_iter is None:
            return None

        data_panned = self.linear_panning(self.sounds[recognition_iter], position_horizontal)

        rate = self.change_freq(position_vertical)

        wavfile.write('output.wav', rate, data_panned)

        subprocess.call(["ffplay", "-nodisp", "-autoexit", 'output.wav'])


    def linear_panning(self, data, position):
        max_val = 32767
        left_channel = data[:, 0] / max_val
        right_channel = data[:, 1] / max_val

        if position == PositionsH.RIGHT.value:
            pan_value = 100
        elif position == PositionsH.LEFT.value:
            pan_value = -100
        else:
            pan_value = 0
        right_amp = pan_value / 200 + 0.5
        left_amp = 1 - right_amp

        sig = np.dstack((left_channel * left_amp, right_channel * right_amp))[0]

        return sig

    def change_freq(self, position):
        if position == PositionV.UP.value:
            rate = self.rate + 20000
        elif position == PositionV.DOWN.value:
            rate = self.rate - 20000
        else:
            rate = self.rate

        return rate
