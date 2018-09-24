from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf
import os
import vggish_input
import vggish_params
import vggish_slim
from pydub import AudioSegment
from audioModel import predict, train
from audioInput import getLaughTracks, getNoise
slim = tf.contrib.slim

if not os.path.isfile('vggish_model.ckpt'):
    wget https://storage.googleapis.com/audioset/vggish_model.ckpt

if not os.path.isfile('vggish_pca_params.npz'):
    wget https://storage.googleapis.com/audioset/vggish_pca_params.npz


flags = tf.app.flags

flags.DEFINE_string(
    'number_of_samples', "5",
    'Number of samples to use')

flags.DEFINE_string(
    'epochs', "5",
    'Number of epochs')

FLAGS = flags.FLAGS

def trainAndSaveAndPredict(test_data, number_of_classes = 2, number_of_samples = 1, epochs = 5, getData = getLaughTracks, use_cache = True, log = True):
    def curriedGetSamples(shuf):
        return getData(number_of_samples = number_of_samples, shuf = shuf, use_cache = use_cache, log = log)
    model_name = 'model_%s_%s' % (number_of_samples, epochs)
    preds = train(curriedGetSamples, number_of_classes, model_name = model_name, epochs = epochs)
    

    return predict(model_name, number_of_classes, test_data)

def printResults(preds, expected = None): 
    with tf.Graph().as_default(), tf.Session() as sess:
        print(preds)
        print(sess.run(tf.argmax(input=preds, axis=1))) 
        print('expected results', expected)

def trainForNoise(number_of_samples=5, epochs=5):
    use_cache = False
    print('training on noise, sin, and constant waves')
    (features, labels) = getNoise(shuf=False, number_of_samples = 2)
    preds = trainAndSaveAndPredict(features, number_of_classes = 3, number_of_samples = number_of_samples, epochs = epochs, getData = getNoise)
    printResults(preds, [0, 0, 1, 1, 2, 2])
    
def trainForLaughter(number_of_samples=5, epochs=5):  
    use_cache = True
    print('training on laughter and not laughter')
    (features, labels) = getLaughTracks(shuf=False, number_of_samples = 2, use_cache = use_cache, use_full_files = False, log=False)
    preds = trainAndSaveAndPredict(features, number_of_classes = 2, number_of_samples = number_of_samples, epochs = epochs, getData = getLaughTracks, use_cache = use_cache, log = False)
    printResults(preds, [0, 0, 1, 1])
    
def main(_):
    trainForLaughter(number_of_samples=int(FLAGS.number_of_samples), epochs=int(FLAGS.epochs))
    

if __name__ == '__main__':
    tf.app.run()
