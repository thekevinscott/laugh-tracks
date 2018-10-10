import sys
sys.path.insert(0, '/ai/laugh-tracks')
from vggish_input import waveform_to_examples
import numpy as np

SAMPLE_RATE = 44100
class AudioData:
    # def __init__(self):

    def normalize(self, arr):
        numerator = arr - np.min(arr)
        divisor = np.max(arr) - np.min(arr)
        if divisor > 0:
            #print('numerator', numerator, 'divisor', divisor)
            assert divisor > 0, "Divisor is 0, Numerator: %f, Divisor: %f" % (numerator, divisor)
            normalized = numerator / divisor
            return normalized

        assert np.max(arr) == 0 and np.min(arr) == 0, "arr has values that are not 0, something is fishy"
        return arr

    def getSamplesAsVggishInput(self, sample):
        vggish_samples = waveform_to_examples(np.array(sample), SAMPLE_RATE)
        return self.normalize(np.array(vggish_samples))


# audioData = AudioData()
