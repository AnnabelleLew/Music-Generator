import os
import numpy as np
import matplotlib.mlab as mlab
import librosa

import scipy
from pathlib import Path
from scipy.ndimage.morphology import generate_binary_structure

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
    hist, bins = np.histogram(np.log(specs[0].flatten()), bins=specs[0].flatten().size//2, density=True)
    cumulative_distr = np.cumsum(hist * np.diff(bins))
    frac_cut = 0.9
    bin_index_of_cutoff = np.searchsorted(cumulative_distr, frac_cut)
    cutoff_log_amplitude = bins[bin_index_of_cutoff]
    audio_data = spsi(local_peaks_v3(np.log(spec), cutoff_log_amplitude), 4096, 4096 // 2)
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
    audio_output = np.zeros(frames * step_size + frame_len - step_size) # stores output
    hanning_window = scipy.signal.hanning(frame_len, sym=True)

    for frame in range(frames):
        for bin in range(1, bins - 1):
            # IDENTIFY PEAKS
            # a peak is a bin where the bins adjacent to it are both smaller
            alpha = spec[bin - 1, frame]
            beta = spec[bin, frame]
            gamma = spec[bin + 1, frame]

            if alpha < beta and gamma < beta:
                # GIVEN A PEAK, ESTIMATE PEAK PHASE RATE
                # uses the formula to calculate shift, then uses this to calculate the frequency of the peak
                if alpha - (2 * beta) + gamma != 0:
                    true_peak_pos = 0.5 * ((alpha - gamma) / (alpha - (2 * beta) + gamma))
                else:
                    true_peak_pos = 0
                freq_at_peak = (2 * np.pi * (bin + true_peak_pos)) / frame_len

                # ACCUMULATE PHASE AT PEAKS
                # update phase values using new frequency
                phase_accum[bin] = phase_accum[bin] + step_size * freq_at_peak
                peak_phase = phase_accum[bin]

                # PHASE LOCK REMAINING BINS TO PEAKS
                # until another peak is reached, shift the remaining bin positions until a trough is reached
                if true_peak_pos < 0:
                    # shift all right bins by pi, and give all left bins the same phase as the peak bin
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
                    # shift all left bins by pi, and give all right bins the same phase as the peak bin
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
        # merges the reconstruction with the current audio
        output_audio[frame * step_size:frame * step_size + frame_len] += construction
    return output_audio

def local_peaks_v3(data, cutoff=0):
    fp = generate_binary_structure(rank=2,connectivity=1)
    threshold = np.zeros(data.shape)
    threshold.fill(cutoff)
    peaks = np.logical_and(True, np.greater(data, threshold)).astype('int')
    return peaks * data
