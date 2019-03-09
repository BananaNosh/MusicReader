import matplotlib.pyplot as plt
from scipy.fftpack import fft as fourier
import scipy
import numpy as np
from scipy.io import wavfile # get the api
import essentia.standard as es
import essentia
import os

if __name__ == '__main__':
    filename = os.path.join("data", "let_it_be.wav")
    fs_rate, signal = wavfile.read(filename)
    print("Frequency sampling", fs_rate)
    channel_count = len(signal.shape)
    print("Channels", channel_count)
    if channel_count == 2:
        signal = signal.sum(axis=1) / 2
    signal = signal[:30*fs_rate]
    sample_size = signal.shape[0]
    print("Complete Samplings N", sample_size)
    track_length_sec = sample_size / float(fs_rate)
    print("secs", track_length_sec)
    sampling_length = 1.0 / fs_rate  # sampling interval in time
    print("Timestep between samples Ts", sampling_length)
    t = np.arange(0, track_length_sec, sampling_length)  # time vector as scipy arange field / numpy.ndarray
    print(t)
    fourier_transformed = abs(fourier(signal))
    fft_left_side = fourier_transformed[:sample_size // 2]  # one side FFT range
    fft_freqs = scipy.fftpack.fftfreq(signal.size, sampling_length)
    fft_freqs_left_side = fft_freqs[:sample_size // 2]  # one side frequency range
    plt.subplot(211)
    p1 = plt.plot(t, signal, "g")  # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    p2 = plt.plot(fft_freqs_left_side, abs(fft_left_side), "b")  # plotting the positive fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.show()

    audio = es.MonoLoader(filename=filename)()
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    print("bpm", bpm)
    bps = 2#bpm / 60
    print("bps", bps)
    hpcps = []
    for b in range(int(track_length_sec * bps)):
        parts = 5
        spectrum = None
        for i in range(parts):
            frame = audio[int(b * fs_rate + i * fs_rate / bps / parts)
                          :int(b * fs_rate + (i+1) * fs_rate / bps / parts)]  # for one beat
            # frame = audio[s *fs_rate: (s+1)* fs_rate]
            if spectrum is None:
                spectrum = es.Spectrum()(frame)
            else:
                spectrum += es.Spectrum()(frame)
        es_frequencies, es_magnitudes = es.SpectralPeaks()(spectrum)
        hpcp = es.HPCP()(es_frequencies, es_magnitudes)
        hpcps.append(hpcp)
    for h in hpcps:
        names = ["a", "b", "h", "c", "cis", "d", "dis", "e", "f", "fis", "g", "gis"]
        print([f"{name}-{v:0.2}" for name, v in zip(names, h) if v > 0.1])
    chords = es.ChordsDetection()(essentia.array(hpcps))
    print(chords)
