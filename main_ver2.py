import numpy
from matplotlib import pyplot, mlab
import scipy.io.wavfile
import os
from collections import defaultdict

SAMPLE_RATE = 44100  # Hz
WINDOW_SIZE = 2048  # размер окна, в котором делается fft
WINDOW_STEP = 512  # шаг окна
WINDOW_OVERLAP = WINDOW_SIZE - WINDOW_STEP


def get_wave_data(wave_filename):
    sample_rate, wave_data = scipy.io.wavfile.read(wave_filename)
    assert sample_rate == SAMPLE_RATE, sample_rate
    if isinstance(wave_data[0], numpy.ndarray):  # стерео
        wave_data = wave_data.mean(1)
    return wave_data


def show_specgram(wave_data):
    fig = pyplot.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.specgram(wave_data,
                NFFT=WINDOW_SIZE, noverlap=WINDOW_OVERLAP, Fs=SAMPLE_RATE)
    pyplot.show()


def get_fingerprint(wave_data):
    # pxx[freq_idx][t] - мощность сигнала
    pxx, _, _ = mlab.specgram(wave_data,
                              NFFT=WINDOW_SIZE, noverlap=WINDOW_OVERLAP, Fs=SAMPLE_RATE)
    band = pxx[15:250]  # наиболее интересные частоты от 60 до 1000 Hz
    return numpy.argmax(band.transpose(), 1)  # max в каждый момент времени


def get_samples(samples_dir):
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


sample_files_dir = "sounds"
samples_fingerprints = get_samples(sample_files_dir)
print(samples_fingerprints)

test_file = "sounds/си.wav"
file_data = get_wave_data(test_file)
test_fingerprint = get_fingerprint(file_data)

best_variant = ""
best_metric = 0
for k, v in samples_fingerprints.items():
    match = get_fingerprint_match(v, test_fingerprint)
    metric = max(match)
    mid = sum(match)/len(match)
    print(k, metric, mid, metric/mid)
    if metric > best_metric:
        best_metric = metric
        best_variant = k
    #compare_fingerprints(v, test_fingerprint, k)
    #compare_fingerprints(test_fingerprint, v, k)
print("Its a " + best_variant)
