import numpy as np

def generate_array(specs):
    frames = np.zeros(len(specs), dtype="int")
    for i in range(len(specs)):
        frames[i] += specs[i].shape[1]
    audio_output = np.zeros((specs[0].shape[0], np.amin(frames)))
    random_seed = np.random.randint(len(specs), size=audio_output.shape)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            audio_output[i][j] += specs[random_seed[i][j]][i][j]
    return audio_output
