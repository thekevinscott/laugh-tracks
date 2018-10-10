import sys
sys.path.insert(0, '/ai/laugh-tracks')
from vggish_input import waveform_to_examples
import numpy as np

SAMPLE_RATE = 44100
class AudioData:
    # def __init__(self):

    def normalize(self, arr):
        numerator = arr - np.min(arr)
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        divisor = arr_max - arr_min
        if divisor > 0:
            #print('numerator', numerator, 'divisor', divisor)
            # assert divisor > 0, "Divisor is 0, Numerator: %f, Divisor: %f" % (numerator, divisor)
            normalized = numerator / divisor
            return normalized

        assert arr_max == arr_min, "arr has min and max that are not equal, min: %f, max: %f" % (arr_min, arr_max)
        arr.fill(0)
        return arr

    def getSamplesAsVggishInput(self, sample):
        vggish_samples = waveform_to_examples(np.array(sample), SAMPLE_RATE)
        return self.normalize(np.array(vggish_samples))


# audioData = AudioData()
