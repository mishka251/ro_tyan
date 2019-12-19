import os
from typing import Tuple, List

import numpy
from numpy import argmax, abs as np_abs
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import scipy.io.wavfile
from notes import notes

FD = 22050  # частота дискретизации, отсчётов в секунду


def get_wave_data(filename: str) -> numpy.array:
    """
    Тут вытаскиваем из файла данные
    :param filename: имя файла
    :return: массив данных
    """
    sample_rate, wave_data = scipy.io.wavfile.read(filename)
    if isinstance(wave_data[0], numpy.ndarray):  # стерео
        wave_data = wave_data.mean(1)
    return wave_data


def get_max_freq(wave_data: numpy.array, name: str = "", use_plot: bool = True) -> float:
    """
    Тут с помощью Фурье вытаскиваем из данных файла самую "сильную" частоту
    :param wave_data: данные файла
    :param name: название(для графика)
    :param use_plot: нужен ли график(да/нет)
    :return: самая сильная частота
    """
    # вычисляем преобразование Фурье. Сигнал действительный, поэтому надо использовать rfft, это быстрее, чем fft
    spectrum: numpy.array = rfft(wave_data)
    N: int = len(wave_data)

    frequency: numpy.array = rfftfreq(N, 1. / FD)
    wtf: numpy.array = np_abs(spectrum) / N

    if use_plot:
        plt.plot(frequency, wtf)
        plt.xlabel('Частота, Гц')
        plt.ylabel('Напряжение, мВ')
        plt.title('Спектр ' + name)
        plt.grid(True)
        plt.show()

    index_max = argmax(wtf)
    return frequency[index_max]


def test_note(wave_data: numpy.array, name: str = "", use_plot: bool = True) -> Tuple[str, float]:
    """
    Распознование одной ноты
    :param wave_data: данные для анализа
    :param name: название(для графика)
    :param use_plot: Надо ли график
    :return: название ноты, частота
    """
    tested_frequency: float = get_max_freq(wave_data, name, use_plot)
    nearest_frequency: float = -10000
    nearest_note_name: str = ""
    for frequency in notes.keys():
        if abs(frequency - tested_frequency) < abs(nearest_frequency - tested_frequency):
            nearest_frequency = frequency
            nearest_note_name = notes[frequency]
    return nearest_note_name, tested_frequency


def test_files() -> None:
    test_dirs: List[str] = ["samples_1", "samples_2", "sounds", "fwd"]#если тестить много файлов сразу - кидайте в папку и сюда эту папку
    for test_dir in test_dirs:
        files: List[str] = os.listdir(test_dir)
        files: List[str] = list(filter(lambda file: file.endswith('.wav'), files))
        print(test_dir)
        for file in files:
            wave_data = get_wave_data(f"{test_dir}/{file}")
            name, frequency = test_note(wave_data)
            print(f"Файл {file} - это нота {name} с частотой {frequency} Гц")
            pass
        print()
        print()
        pass


def test_new_file():
    file_path: str = "sounds/до.wav"#сюда подставить путь до файла который тестить
    file_data = get_wave_data(file_path)
    name, freq = test_note(file_data, file_path, True)
    print(f"Файл {file_path} - это нота {name} с частотой {freq} Гц")


test_files()#эта строчка запускает тестовые файлы
# test_new_file() #раскомментарьте эту строчку для запуска теста одного новго файла(и см строка69)
