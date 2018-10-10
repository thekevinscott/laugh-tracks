from AudioData import AudioData
import numpy as np

def test_it_handles_already_normalized_data():
    audioData = AudioData()
    arr = np.array([0, 1])
    assert (audioData.normalize(arr) == arr).all()

def test_that_it_normalizes_data():
    audioData = AudioData()
    arr = np.array([0, 2])
    assert (audioData.normalize(arr) == np.array([0, 1])).all()

def test_that_it_handles_negative_numbers():
    audioData = AudioData()
    arr = np.array([-1, 0])
    assert (audioData.normalize(arr) == np.array([0, 1])).all()

def test_that_it_handles_a_large_range():
    audioData = AudioData()
    arr = np.array([-2, 2])
    assert (audioData.normalize(arr) == np.array([0, 1])).all()

def test_that_it_handles_multiple_numbers():
    audioData = AudioData()
    arr = np.array([-2, -1, 0, 1, 2, 3])
    assert (audioData.normalize(arr) == np.array([0, 0.2, 0.4, 0.6, 0.8, 1])).all()

def test_that_it_handles_multiple_dimensions():
    audioData = AudioData()
    arr = np.array([[0, 1], [3, 4]])
    assert (audioData.normalize(arr) == np.array([[0, 0.25], [0.75, 1]])).all()

def test_that_it_handles_multiple_dimensions_with_negative_numbers():
    audioData = AudioData()
    arr = np.array([[-2, -3], [2, 1]])
    assert (audioData.normalize(arr) == np.array([[0.2, 0], [1, 0.8]])).all()

def test_that_it_handles_a_divisor_that_is_zero():
    audioData = AudioData()
    arr = np.array([0, 0])
    assert (audioData.normalize(arr) == np.array([0, 0])).all()

def test_that_it_handles_a_divisor_that_is_zero_with_numbers_that_are_not():
    audioData = AudioData()
    arr = np.array([2, 2])
    assert (audioData.normalize(arr) == np.array([0, 0])).all()
