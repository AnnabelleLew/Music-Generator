import os
import numpy as np
import matplotlib.mlab as mlab
import librosa

import scipy
from pathlib import Path
import IPython

""" Converts a folder of .wav songs into spectrogram data.

        This runs through each .wav song in the folder and converts it into a
        spectrogram, which is stored in a list.

        Parameters
        ----------
        dir : String
            The filepath to the folder where the songs are stored.

        Returns
        -------
        List
            Returns a list of spectrograms, which has as many songs as there
            are in the folder. Each spectrogram has dimensions of shape
            (NFFT, T), where T is the sampling rate * the number of seconds in
            the song."""
def songs_to_arrays(dir):
    spectrograms = list() # stores spectrograms
    # goes through directory
    for song in os.listdir(dir):
        # ignores .DS_Store (and eliminates all associated errors)
        if song == ".DS_Store":
            continue
        # loads each song, and converts into a spectrogram to add to the list
        audio_data, sampling_rate = librosa.load(Path(dir + song), sr=1, mono=True)
        audio_data = (audio_data*(2**15)).astype('int16')
        spec, freqs, t = mlab.specgram(audio_data, NFFT=512, Fs=sampling_rate,
                                       window=mlab.window_hanning,
                                       noverlap=512 // 8)
        spectrograms.append(spec)
    return spectrograms

""" Converts a given spectrogram back into a .wav file.

        This takes a spectrogram, clears noise, roughly guesses what each sound
        was before it was put into the frequency bins, and saves it into a .wav
        file.

        Parameters
        ----------
        spec : np.ndarray
            The spectrogram data, of shape (F, T), where F is the number of
            frequency bins and T is the number of time bins.
        filepath : str
            The filepath where the audio data is saved to.
        vol : int
            The volume. Multiplies all amplitudes in the audio data by 10e(vol).
        frac_cut : float
            What fraction of the spectrogram data to remove when clearing noise.
            Defaulted to remove the 90% quietest frequencies, and only keep the
            10% loudest frequencies.

        Returns
        -------
        np.ndarray
            Returns 1D audio data, which can be played using IPython."""
def arrays_to_songs(spec, filepath, vol=4, frac_cut=0.9):
    spec += 1e-20 # ensures all values are above zero (deals with bugs)
    # gets rid of noise
    hist, bins = np.histogram(np.log(spec.flatten()), bins=spec.flatten().size//2, density=True)
    cumulative_distr = np.cumsum(hist * np.diff(bins))
    bin_index_of_cutoff = np.searchsorted(cumulative_distr, frac_cut)
    cutoff_log_amplitude = bins[bin_index_of_cutoff]
    # converts spectrogram back into audio data that can be processed into a .wav file
    audio_data = spsi(local_peaks_v3(np.log(spec), cutoff_log_amplitude), 512, 256 // 8, vol)
    scipy.io.wavfile.write(filepath, 1, audio_data) # you can customize filepath!
    return audio_data

""" Method that converts spectrogram data back into a .wav file.

        This uses the single-pass spectrogram inversion (SPSI) algorithm, which
        identifies peaks in the spectrogram, estimates their true frequency, and
        adds them to a phase accumulator. The phases at the remaining bins are
        then determined based on this. This data is then combined with the
        magnitudes of the original spectrogram data to compute the IFFT, which
        are Hanning-windowed to produce actual audio data.

        THIS CONCEPT IS NOT MINE. BASED ON THIS PAPER:
        Beauregard, G., Harish, M. and Wyse, L. (2015), Single Pass Spectrogram
        Inversion, in Proceedings of the IEEE International Conference on
        Digital Signal Processing. Singapore, 2015.

        Parameters
        ----------
        spec : np.ndarray
            The spectrogram data.
        frame_len : int
            The analysis frame window length.
        step_size : int
            The analysis hop size.
        vol : int
            The volume.

        Returns
        -------
        np.ndarray
            Returns 1D audio data."""
def spsi(spec, frame_len, step_size, vol=4):
    spec = np.abs(spec) # gets the magnitude of the spectrogram data
    bins, frames = spec.shape # the number of frequency bins and time frames
    phase_accum = np.zeros(bins) # stores phase values in a given bin
    output_audio = np.zeros(frames * step_size + frame_len - step_size) # stores output
    hanning_window = scipy.signal.hann(frame_len, sym=True)

    for frame in range(frames):
        # gets history (based on Griffin & Lim algorithm)
        for bin in range(1, bins - 1):
            # IDENTIFY PEAKS
            # a peak is a bin where the bins adjacent to it are both smaller
            alpha = spec[bin - 1, frame]
            beta = spec[bin, frame]
            gamma = spec[bin + 1, frame]
            if alpha < beta and gamma < beta: # checks for a peak
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
        construction = (np.real(np.fft.ifft(construction)) * hanning_window * (10 ** vol)).astype('int')

        # OVERLAP
        # merges the reconstruction with the current audio
        output_audio[frame * step_size:frame * step_size + frame_len] += construction
    return output_audio

""" Gets the peaks given spectrogram data.

        This takes a spectrogram and the frequency bins with the highest amplitudes,
        and only keeps the data at those bins.

        Parameters
        ----------
        data : np.ndarray
            The spectrogram data.
        cutoff : np.ndarray
            The highest amplitude frequencies.

        Returns
        -------
        np.ndarray
            Returns only the peaks of the data, and zeros for the rest."""
def local_peaks_v3(data, cutoff=0):
    # creates cutoff data based off of cutoff frequencies
    threshold = np.zeros(data.shape)
    threshold.fill(cutoff)
    peaks = np.logical_and(True, np.greater(data, threshold)) # only keeps values above the cutoff data
    return peaks * data # keeps all data above cutoff, and zeros all other data.
