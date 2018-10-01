from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
import os
import vggish_input
import vggish_params
import vggish_slim
import math
from pydub import AudioSegment
from audioUtils import readFolder, readFolderRecursive, load
import hashlib
import pprint, pickle

slim = tf.contrib.slim

SAMPLE_RATE = 44100

def gatherTrainingData(files):
    audioFiles = []
    for file in files:
        audio = load(file)

        audioFiles.append({
            'audio': audio,
            'file': file,
        })
        
    return audioFiles

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
        random.shuffle(labeled_examples)

    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)

def calculateMsForChunks(chunks):
    return int(960*chunks) + 16

def calculateSamplesForMs(ms):
    return int(SAMPLE_RATE/1000*ms)


def calculateChunksForMs(ms):
    return math.floor((ms - 16) / 960)

def calculateMsForSamples(samples):
    return samples * 1000 / SAMPLE_RATE

def calculateChunksForSamples(samples):
    return calculateChunksForMs(calculateMsForSamples(samples))



def getHashForFiles(files, extra_string = ''):
    hash_object = hashlib.md5(('%s' % extra_string).join(files).encode())
    return '%s' % (hash_object.hexdigest())

def getFilePathsForClass(c):
    files = readFolderRecursive('samples/%s' % (c), r"(.*)\.wav$")
    return files
            
def getSamplesForFile(file, seconds = None):
    try:
        print('1', file)
        audio = load(file)
        print('2')        
        start = random.randint(0, SAMPLE_RATE * .96)
        print('start', start)
        samples = audio.get_array_of_samples()
        print('3')
        return samples[start:]
    except:
        print('file failed', file)
        raise Exception('file failed: %s' % file)        

# accepts a numpy array representing a single audio file, or multiple files concat'ed together
def getSamplesAsVggishInput(sample):
    x = np.array(vggish_input.waveform_to_examples(np.array(sample), SAMPLE_RATE))
    numerator = x-np.min(x)
    divisor = np.max(x)-np.min(x)
    print('numerator', numerator, 'divisor', divisor)
    assert divisor > 0, "Divisor is 0, Numerator: %f, Divisor: %f" % (numerator, divisor) 
    normalized = numerator/divisor
    return normalized
    #print('n', normalized) 
    #return x


def getChunksForSamples(samples, file):
    chunks = []
    for i in range(0, getChunksForMs(getMsForSamples(len(samples)))):
        chunks.append({
            'file': file,
            'start': i,
        })
        
    return chunks

def getSamplesForFiles(files):
    fileSamples = []
    for file in files:      
        samples = getSamplesForFile(file)
        chunks = getChunksForSamples(samples, file)
        fileSample = {
            'chunks': chunks,
            'samples': samples
        }
        fileSamples.append(fileSample)
              
    return fileSamples


def getSamplesForFileWithOptionalCache(files, number_of_samples, log=False, use_cache = True):  
    if use_cache == False:
        return getSamplesForFiles(files, number_of_samples, log=log)
    
    filename = getHashForFiles(files, number_of_samples)
    #print('using cache with file', filename)
    cache_file = 'cache/%s.pkl' % filename
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if not os.path.isfile(cache_file):
        print('no cache file available, building one')
        samples, chunks = getSamplesForFiles(files, number_of_samples, log=log)
        output = open(cache_file, 'wb')
        data = (samples, chunks)
        pickle.dump(data, output)
        output.close()

    pkl_file = open(cache_file, 'rb')
    (samples, chunks) = pickle.load(pkl_file)
    pkl_file.close()
    return samples, chunks

def getData(files, arr):
    fileSamples = getSamplesForFile(files)
    zippedUp = []
    for file in fileSamples:
        samples = file['samples']
        chunks = file['chunks']
        labels = np.array([arr] * chunks.shape[0])
        zippedUp.append({
            'samples': samples,
            'chunks': chunks,
            'labels': labels,
        })

    return zippedUp

def getOneHot(class_num, idx):
    arr = np.zeros(class_num)
    arr[idx] = 1
    return arr

def processWavFile(file):
    samples, chunks = getSamplesForFileWithOptionalCache([file], None)
    return samples

def getSamples(classes, shuf = True):
    exes = []
    whys = []
    allChunks = []
    foundFiles = {}
    clsLengths = {}
    
    for idx, cls in enumerate(classes):
        files = getFilePathsForClass(cls)
        foundFiles[cls] = files
        filesData = getData(files, getOneHot(len(classes), idx))
        for file in filesData:
            x = file['samples']
            y = file['labels']
            chunks = file['chunks']
            clsLengths[cls] = len(x)
            allChunks.append(chunks)
            exes.append(x)
            whys.append(y)
        
    print('Number of features per class', clsLengths)
    features = np.concatenate(exes)
    labels = np.concatenate(whys)
    concatChunks = np.concatenate(allChunks)
    if shuf == True:
        return shuffleSamples(features, labels, concatChunks)
    return (features, labels, concatChunks)


def shuffleSamples(features, labels, concatChunks):
    labeled_examples = list(zip(features, labels, concatChunks))
    new_examples = random.sample(labeled_examples, len(labeled_examples))
    features = [example for (example, _, _) in new_examples]
    labels = [label for (_, label, _) in new_examples]
    chunks = [chunk for (_, _, chunk) in new_examples]
    return (features, labels, chunks)

def getLaughTracks(shuf = True, split = 0.2):
    return getSamples(['laughter', 'notlaughter'], shuf = shuf)
