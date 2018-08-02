""" TO DO:
    - get random size for array
    - get probability of different values for array
    - generate random array
"""
import numpy as np

def generate_array(specs):
    frames = np.zeros(len(specs), dtype="int")
    for i in range(len(specs)):
        frames[i] += specs[i].shape[1]
    audio_output = np.zeros((specs[0].shape[0], np.random.choice(frames)))
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            audio_output += np.random.choice([specs[k][i][j] for k in range(len(specs)) if j < len(specs[k][i])])
    return audio_output
