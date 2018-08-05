import numpy as np

""" Generates a random spectrogram.

        This takes in a list of spectrograms, and returns a randomly generated
        spectrogram based off of this data.

        Parameters
        ----------
        specs : List
            A list of spectrograms, each of shape (F, T), where F is the number
            of frequency bins and T is the number of time bins.

        Returns
        -------
        np.ndarray
            Returns a np.ndarray of shape (F, T), where F is the number of
            frequency bins, and T is the smallest time bin among the spectrograms.
            This is randomly generated!"""
def generate_array(specs):
    frames = np.zeros(len(specs), dtype="int") # stores the possible different number of time frames
    for i in range(len(specs)):
        frames[i] += specs[i].shape[1]
    audio_output = np.zeros((specs[0].shape[0], np.amin(frames))) # spectrogram, only gets minimum time frame to eliminate errors
    random_seed = np.random.randint(len(specs), size=audio_output.shape) # randomly generates which spectrograms to take audio from
    # for each frequency bin and time frame, add the amplitude at that location from a random spectrogram
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            audio_output[i][j] += specs[random_seed[i][j]][i][j]
    return audio_output
