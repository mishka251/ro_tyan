#!/usr/bin/env python
# coding=utf8
import os
from typing import Dict

import numpy
from numpy import argmax, abs as np_abs
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import scipy.io.wavfile
from notes import notes

FD = 22050  # частота дискретизации, отсчётов в секунду


def get_wave_data(filename):
    sample_rate, wave_data = scipy.io.wavfile.read(filename)
    if isinstance(wave_data[0], numpy.ndarray):  # стерео
        wave_data = wave_data.mean(1)
    return wave_data


def get_max_freq(wave_data, name="", use_plot=True):
    sig = wave_data
    # вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft
    spectrum = rfft(sig)
    N = len(sig)

    herz = rfftfreq(N, 1. / FD)
    wtf = np_abs(spectrum) / N
    if use_plot:
        plt.plot(herz, wtf)
        plt.xlabel('Частота, Гц')
        plt.ylabel('Напряжение, мВ')
        plt.title('Спектр ' + name)
        plt.grid(True)
        plt.show()

    index_max = argmax(wtf)
    max_herz = herz[index_max]

    return max_herz


def test_note(wave_data, name="", use_plot=True):
    max_freq = get_max_freq(wave_data, name, use_plot)
    best_key = -10000
    best_val = ""
    for key in notes.keys():
        if (abs(key - max_freq) < abs(best_key - max_freq)):
            best_key = key
            best_val = notes[key]
    return best_val, max_freq


test_dirs = ["samples_1", "samples_2", "sounds", "fwd"]
for test_dir in test_dirs:
    files = os.listdir(test_dir)
    files = list(filter(lambda file: file.endswith('.wav'), files))
    print(test_dir)
    for file in files:
        wave_data = get_wave_data(f"{test_dir}/{file}")
        value, max_freq = test_note(wave_data)
        print(f"{file} freq = {max_freq}- is {value}")
        pass
    print()
    print()
    pass


