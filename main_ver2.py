from typing import Dict

import numpy
from matplotlib import pyplot, mlab
import scipy.io.wavfile
import os
from collections import defaultdict
import random
from typing.io import TextIO

SAMPLE_RATE = 44100  # Hz
WINDOW_SIZE = 2048  # размер окна, в котором делается fft
WINDOW_STEP = 512  # шаг окна
WINDOW_OVERLAP = WINDOW_SIZE - WINDOW_STEP


def get_wave_data(wave_filename):
    sample_rate, wave_data = scipy.io.wavfile.read(wave_filename)
    #assert sample_rate == SAMPLE_RATE, sample_rate
    if isinstance(wave_data[0], numpy.ndarray):  # стерео
        wave_data = wave_data.mean(1)
    return sample_rate, wave_data


def show_specgram(sample_rate, wave_data):
    fig = pyplot.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.specgram(wave_data,
                NFFT=WINDOW_SIZE, noverlap=WINDOW_OVERLAP, Fs=sample_rate)
    pyplot.show()


def get_fingerprint(sample_rate, wave_data):
    # pxx[freq_idx][t] - мощность сигнала
    pxx, _, _ = mlab.specgram(wave_data,
                              NFFT=WINDOW_SIZE, noverlap=WINDOW_OVERLAP, Fs=sample_rate)
    band = pxx[15:250]  # наиболее интересные частоты от 60 до 1000 Hz
    return numpy.argmax(band.transpose(), 1)  # max в каждый момент времени


def get_samples(samples_dir)->Dict:
    sample_files = os.listdir(samples_dir)
    sample_files = list(filter(lambda file: file.endswith('.wav'), sample_files))
    # print(sample_files)
    fingerprints = list(map(lambda file: get_wave_data(samples_dir + "/" + file), sample_files))
    samples_fingerprints = dict(zip(sample_files, fingerprints))
    return samples_fingerprints


def get_fingerprint_match(fp1, fp2):
    base_fp_hash = defaultdict(list)
    for time_index, freq_index in enumerate(fp1):
        base_fp_hash[freq_index].append(time_index)
    matches = [t - time_index  # разницы времен совпавших частот
               for time_index, freq_index in enumerate(fp2)
               for t in base_fp_hash[freq_index]]
    return matches


def compare_fingerprints(base_fp, fp, title):
    base_fp_hash = defaultdict(list)
    for time_index, freq_index in enumerate(base_fp):
        base_fp_hash[freq_index].append(time_index)
    matches = [t - time_index  # разницы времен совпавших частот
               for time_index, freq_index in enumerate(fp)
               for t in base_fp_hash[freq_index]]
    pyplot.clf()
    pyplot.hist(matches, 1000)
    pyplot.title(title)
    pyplot.show()


def plot_match(matches, name1, name2):
    pyplot.clf()
    pyplot.hist(matches, 1000)
    pyplot.title(name1+"-"+name2)
    pyplot.show()


def test_file(samples:Dict, test_file:str, test_name:str, log_file:TextIO):
    sample_rate, wave_data = get_wave_data(test_file)
    test_fingerprint = get_fingerprint(sample_rate, wave_data)
    log_file.write("Testing "+test_name+"\n")
    print("Testing " + test_name)
    log_file.write("fingerprint\n")
    log_file.write(str(test_fingerprint))
    log_file.write("\n")
    best_variant = ""
    best_metric = -10000000
    for k, v in samples.items():
        match = get_fingerprint_match(v[1], test_fingerprint)
        _max = max(match)
        mid = sum(match) / len(match)
        metric = mid
        log_file.write(f"{k}, {_max}, {mid}, {metric}\n")
        #plot_match(match, test_name, k)
        if metric > best_metric:
            best_metric = metric
            best_variant = k
    best_random_item = random.randint(0, len(list(samples.keys())))
    best_variant = list(samples.keys())[best_random_item]
    print("Its a " + best_variant)
    log_file.write("Its a " + best_variant)


def test(samples:Dict):
    log: TextIO = open("log.txt", mode="w")
    test_dirs = ["fwd", "samples_1", "samples_2"]
    for test_dir in test_dirs:
        files = os.listdir(test_dir)
        files = list(filter(lambda file: file.endswith('.wav'), files))
        for file in files:
            name, _ = file.split('.wav')
            test_file(samples, test_dir+"/"+file, name, log)
            print("\n")
            log.write("\n\n")
            pass
        pass
    log.close()
    pass


sample_files_dir = "samples_2"
samples_fingerprints = get_samples(sample_files_dir)
test(samples_fingerprints)
