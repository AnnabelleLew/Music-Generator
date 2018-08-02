import os
import numpy as np
import matplotlib.mlab as mlab

import scipy
from pathlib import Path

def songs_to_arrays(dir):
    spectrograms = list()
    for song in os.listdir(dir):
        if song == ".DS_Store":
            continue
        audio_data, sampling_rate = librosa.load(Path(dir + song), sr=44100, mono=True)
        audio_data = (audio_data*(2**15)).astype('int16')
        spec, freqs, t = mlab.specgram(audio_data, NFFT=4096, Fs=sampling_rate,
                                       window=mlab.window_hanning,
                                       noverlap=4096 // 2)
        spectrograms.append(spec)
    return spectrograms

def arrays_to_songs(spec):
    audio_data = spsi(spec, 4096, 4096 // 2)
    return audio_data

"""
THIS CONCEPT IS NOT MINE
Beauregard, G., Harish, M. and Wyse, L. (2015), Single Pass Spectrogram Inversion,
in Proceedings of the IEEE International Conference on Digital Signal Processing. Singapore, 2015.
"""
def spsi(spec, frame_len, step_size):
    spec = np.abs(spec)
    bins, frames = spec.shape # the number of frequency bins and time frames
    phase_accum = np.zeros(bins) # stores phase values in a given bin
    output_audio = np.zeros(frames * step_size + frame_len - step_size) # stores output
    hanning_window = scipy.signal.hann(frame_len, sym=True)

    for frame in range(frames):
        for bin in range(1, bins - 1):
            # IDENTIFY PEAKS
            alpha = spec[bin - 1, frame]
            beta = spec[bin, frame]
            gamma = spec[bin + 1, frame]

            if alpha < beta and gamma < beta:
                # GIVEN A PEAK, ESTIMATE PEAK PHASE RATE
                if alpha - (2 * beta) + gamma != 0:
                    true_peak_pos = 0.5 * ((alpha - gamma) / (alpha - (2 * beta) + gamma))
                else:
                    true_peak_pos = 0
                freq_at_peak = (2 * np.pi * (bin + true_peak_pos)) / frame_len

                # ACCUMULATE PHASE AT PEAKS
                phase_accum[bin] = phase_accum[bin] + step_size * freq_at_peak
                peak_phase = phase_accum[bin]

                # PHASE LOCK REMAINING BINS TO PEAKS
                if true_peak_pos < 0:
                    phase_accum[bin - 1] = peak_phase + np.pi
                    for b in range(bin + 1, bins):
                        if phase_accum[b] < phase_accum[b - 1]:
                            break
                        else:
                            phase_accum[b] = peak_phase + np.pi
                    for b in np.arange(2, bin - 1)[::-1]:
                        if phase_accum[b] < phase_accum[b + 1]:
                            break
                        else:
                            phase_accum[b] = peak_phase
                else:
                    phase_accum[bin + 1] = peak_phase + np.pi
                    for b in range(bin - 1, bins):
                        if phase_accum[b] < phase_accum[b + 1]:
                            break
                        else:
                            phase_accum[b] = peak_phase + np.pi
                    for b in np.arange(2, bin - 1)[::-1]:
                        if phase_accum[b] < phase_accum[b - 1]:
                            break
                        else:
                            phase_accum[b] = peak_phase

        # MERGE PHASES FOR EACH BIN WITH FREQUENCIES, AND DO IFFT
        new_phase = spec[:, frame] * np.exp(1j * phase_accum)
        new_phase[0] = 0
        new_phase[bins - 1] = 0
        construction = np.concatenate([new_phase, np.flip(np.conjugate(new_phase[1:bins - 1]), 0)])
        construction = np.real(np.fft.ifft(construction)) * hanning_window

        # OVERLAP
        output_audio[frame * step_size:frame * step_size + frame_len] += construction
    return output_audio * 10e99
