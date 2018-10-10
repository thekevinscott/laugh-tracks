from AudioData import AudioData
import numpy as np

def test_it_handles_already_normalized_data():
    audioData = AudioData()
    arr = np.array([0, 1])
    assert audioData.normalize(arr) == arr
