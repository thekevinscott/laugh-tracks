from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf
import os
import vggish_input
import vggish_params
import vggish_slim
import math
from pydub import AudioSegment
from audioUtils import readFolder

slim = tf.contrib.slim

SAMPLE_RATE = 44100

def getNoise(shuf = True, number_of_samples = 1, log=False):
    """Returns a shuffled batch of examples of all audio classes.

    Note that this is just a toy function because this is a simple demo intended
    to illustrate how the training code might work.

    Returns:
    a tuple (features, labels) where features is a NumPy array of shape
    [batch_size, num_frames, num_bands] where the batch_size is variable and
    each row is a log mel spectrogram patch of shape [num_frames, num_bands]
    suitable for feeding VGGish, while labels is a NumPy array of shape
    [batch_size, num_classes] where each row is a multi-hot label vector that
    provides the labels for corresponding rows in features.
    """
    # Make a waveform for each class.
    num_seconds = number_of_samples
    sr = 44100  # Sampling rate.
    t = np.linspace(0, num_seconds, int(num_seconds * sr))  # Time axis.
    # Random sine wave.
    freq = np.random.uniform(100, 1000)
    sine = np.sin(2 * np.pi * freq * t)
    # Random constant signal.
    magnitude = np.random.uniform(-1, 1)
    const = magnitude * t
    # White noise.
    noise = np.random.normal(-1, 1, size=t.shape)

    # Make examples of each signal and corresponding labels.
    # Sine is class index 0, Const class index 1, Noise class index 2.
    sine_examples = vggish_input.waveform_to_examples(sine, sr)
    sine_labels = np.array([[1, 0, 0]] * sine_examples.shape[0])
    const_examples = vggish_input.waveform_to_examples(const, sr)
    const_labels = np.array([[0, 1, 0]] * const_examples.shape[0])
    noise_examples = vggish_input.waveform_to_examples(noise, sr)
    noise_labels = np.array([[0, 0, 1]] * noise_examples.shape[0])

    # Shuffle (example, label) pairs across all classes.
    all_examples = np.concatenate((sine_examples, const_examples, noise_examples))
    all_labels = np.concatenate((sine_labels, const_labels, noise_labels))
    labeled_examples = list(zip(all_examples, all_labels))
    if shuf:
        shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)

def getFilePathsForClass(c):
    files = readFolder('samples/%s' % (c))
    collected_files = []
    for file in files:
        path = 'samples/%s/%s' % (c, file)
        #print(path)
        collected_files.append(path)
    return collected_files
            
def getSampleForFile(file, seconds):
    audio = AudioSegment.from_file(file).get_array_of_samples()[0:round(SAMPLE_RATE * seconds)]
    return audio

# accepts a numpy array representing a single audio file, or multiple files concat'ed together
def getFileAsVggishInput(sample):
    return vggish_input.waveform_to_examples(sample, SAMPLE_RATE)

# append every audio file into one enormous massive audio file
def getSamplesForFiles(files, number_of_samples, log=False):
    sample = np.array([])
    if log:
        print('number of samples', number_of_samples, 'reading %i files' % (len(files)), files[0:3])
        
    for file in files:
        current_sample_size = math.ceil(sample.shape[0] / SAMPLE_RATE)
        if current_sample_size < number_of_samples:
            remaining_samples = number_of_samples - math.ceil(sample.shape[0] / SAMPLE_RATE)
            audio = getSampleForFile(file, remaining_samples)
            sample = np.append(sample, audio)
            #print(sample.shape[0] / SAMPLE_RATE)
        
    return getFileAsVggishInput(sample)

def getData(files, number_of_samples, shuf, log, arr):
    if shuf:
        shuffle(files)
    examples = getSamplesForFiles(files, number_of_samples, log)
    labels = np.array([arr] * examples.shape[0])
    
    return (examples, labels)

def getOneHot(class_num, idx):
    arr = np.zeros(class_num)
    arr[idx] = 1
    return arr

def processWavFile(file, log = True):
    return getSamplesForFiles([file], 99999999, log = log)

def getSamples(classes, shuf = True, number_of_samples = None, log=False):
    exes = []
    whys = []
    #print('collecting samples')
    for idx, cls in enumerate(classes):
        files = getFilePathsForClass(cls)
        x, y = getData(files, number_of_samples, shuf, log, getOneHot(len(classes), idx))
        exes.append(x)
        whys.append(y)
    
    all_examples = np.concatenate(exes)
    all_labels = np.concatenate(whys)
    labeled_examples = list(zip(all_examples, all_labels))
    if shuf:
        shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)

def getLaughTracks(number_of_samples = 1, shuf = True, log=True):
    features_name = 'checkpoints/features_%s.npy' % (number_of_samples)
    labels_name = 'checkpoints/labels_%s.npy' % (number_of_samples)
    
    (features, labels) = getSamples(['laughter', 'notlaughter'], shuf = shuf, number_of_samples = number_of_samples, log=log)

    labeled_examples = list(zip(features, labels))
    if shuf:
        shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)