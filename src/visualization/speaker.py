import numpy as np
from scipy.io import wavfile
import glob
import IPython.display as ipd
import sounddevice as sd
import time
import pyaudio


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

    def say_recognition(self, recognition_iter):

        if recognition_iter is None:
            return None

        data_panned = self.linear_panning(self.sounds[recognition_iter])

        rate = 44100

        p = pyaudio.PyAudio()


        # sd.play(data_panned, rate, blocking=True)
        # time.sleep(2.0)
        # sd.stop()
        # status = sd.wait()


        # ipd.Audio('/home/ybaa/Documents/test.wav')

        # ipd.Audio(self.sounds[7], 44100)

        # wavfile.write('output.wav', 44100, data_panned)


    def linear_panning(self, data):
        max_val = 32768
        left_channel = data[:, 0] / max_val
        right_channel = data[:, 1] / max_val

        pan_value = 100
        right_amp = pan_value / 200 + 0.5
        left_amp = 1 - right_amp

        sig = np.dstack((left_channel * left_amp, right_channel * right_amp))[0]

        return sig
