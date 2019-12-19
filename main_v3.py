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


def get_max_freq(wave_data):
    sig = wave_data
    # вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft
    spectrum = rfft(sig)
    N = len(sig)

    herz = rfftfreq(N, 1. / FD)
    wtf = np_abs(spectrum) / N

    index_max = argmax(wtf)
    max_herz = herz[index_max]

    return max_herz


def test_note(wave_data):
    max_freq = get_max_freq(wave_data)
    best_key = -10000
    best_val = ""
    for key in notes.keys():
        if (abs(key - max_freq) < abs(best_key - max_freq)):
            best_key = key
            best_val = notes[key]
    return best_val


test_dirs = ["samples_1", "samples_2"]
for test_dir in test_dirs:
    files = os.listdir(test_dir)
    files = list(filter(lambda file: file.endswith('.wav'), files))
    for file in files:
        wave_data = get_wave_data(f"{test_dir}/{file}")
        value = test_note(wave_data)
        print(f"{file} - is {value}")
        pass
    pass
    #     herz, wtf = get_spectere(wave_data)
    #
    #     index_max = argmax(wtf)
    #     max_herz = herz[index_max]
    #     print(f"{file} max herz = {max_herz}")
    #     plt.plot(herz, wtf)
    #     plt.xlabel('Частота, Гц')
    #     plt.ylabel('Напряжение, мВ')
    #     plt.title('Спектр '+file)
    #     plt.grid(True)
    #     plt.show()
    #     pass
    # pass

# а можно импортировать numpy и писать: numpy.fft.rfft

# а это значит, что в дискретном сигнале представлены частоты от нуля до 11025 Гц (это и есть теорема Котельникова)
# N = 2000 # длина входного массива, 0.091 секунд при такой частоте дискретизации
# сгенерируем сигнал с частотой 440 Гц длиной N
# pure_sig = array([6.*sin(2.*pi*440.0*t/FD) for t in range(N)])
# сгенерируем шум, тоже длиной N (это важно!)
# noise = uniform(-50.,50., N)
# суммируем их и добавим постоянную составляющую 2 мВ (допустим, не очень хороший микрофон попался. Или звуковая карта или АЦП)
# sig = pure_sig + noise + 2.0 # в numpy так перегружена функция сложения
# print(sig)
# wave_filename="samples_2/до.wav"
# #wav = wave.open(, mode="r")
# sample_rate, wave_data = scipy.io.wavfile.read(wave_filename)
# if isinstance(wave_data[0], numpy.ndarray):  # стерео
#     wave_data = wave_data.mean(1)
# sig = wave_data
# # вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft
# spectrum = rfft(sig)
# N = len(sig)
# # нарисуем всё это, используя matplotlib
# # Сначала сигнал зашумлённый и тон отдельно
# plt.plot(arange(N)/float(FD), sig) # по оси времени секунды!
# #plt.plot(arange(N)/float(FD), pure_sig, 'r') # чистый сигнал будет нарисован красным
# plt.xlabel(u'Время, c') # это всё запускалось в Python 2.7, поэтому юникодовские строки
# plt.ylabel(u'Напряжение, мВ')
# plt.title(u'Зашумлённый сигнал и тон 440 Гц')
# plt.grid(True)
# plt.show()
# # когда закроется этот график, откроется следующий
# # Потом спектр
# herz = rfftfreq(N, 1./FD)
# plt.plot(herz, np_abs(spectrum)/N)
# # rfftfreq сделает всю работу по преобразованию номеров элементов массива в герцы
# # нас интересует только спектр амплитуд, поэтому используем abs из numpy (действует на массивы поэлементно)
# # делим на число элементов, чтобы амплитуды были в милливольтах, а не в суммах Фурье. Проверить просто — постоянные составляющие должны совпадать в сгенерированном сигнале и в спектре
# plt.xlabel(u'Частота, Гц')
# plt.ylabel(u'Напряжение, мВ')
# plt.title(u'Спектр')
# plt.grid(True)
# plt.show()
