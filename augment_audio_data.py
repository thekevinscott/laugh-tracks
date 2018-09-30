# A function to augment data
# --input_dir
# --output_dir
# types of augmentation:
# - changing the volume
# - adding compression
# - adding noise to laughter
# future augmentation:
# - adding noise
# - adding eq

from __future__ import print_function

from audioUtils import stripSilenceInPlace
import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import subprocess
import numpy as np
import math
import os
import IPython.display as ipd
from audioUtils import readFolder

from pydub.effects import compress_dynamic_range



flags = tf.app.flags

flags.DEFINE_string('input', None, 'The input directory containing the audio samples')
flags.DEFINE_string('output', None, 'The output directory to write the augmented files to')
flags.DEFINE_string('mix_dir', None, 'The directory to include for mixing files')
flags.DEFINE_string('number_of_transforms', '5', 'Number of times to transform file')
FLAGS = flags.FLAGS

def randF(min, max, divisor = 1000):
    r = random.randint(int(min * divisor), int(max * divisor))
    return r / divisor

def augmentSound(file, transform):
    sound = AudioSegment.from_file(file)
    sound = transform(sound)
    return sound
    #display(ipd.Audio(target))

# TRANSFORMS
def changeGain(sound, max_amount_of_gain=50, makeup = 0):
    gain = random.randint(0, max_amount_of_gain)
    return sound.apply_gain(-sound.max_dBFS - gain + makeup)

def addCompression(sound):
    threshold = randF(-40, 0)
    ratio = randF(0, 10)
    attack = randF(0, 10)
    release = randF(0, 500)
    return compress_dynamic_range(sound, threshold=threshold, ratio=ratio, attack=attack, release=release)

def mixWith(noises):
    def curriedMixWith(sound):
        max = 10
        noise = changeGain(AudioSegment.from_file(random.choice(noises)), max, int(max / 2))
        return changeGain(sound, max, int(max / 2)).overlay(noise)
    return curriedMixWith

noise_files = readFolder(FLAGS.mix_dir)
noise = []
for file in noise_files:
    noise.append('%s/%s' % (FLAGS.mix_dir, file))

curriedMixWith = mixWith(noise)

TRANSFORMS = {
    'changeGain': changeGain,
    'addCompression': addCompression,
    'mixWith': curriedMixWith,
    'mixWith': curriedMixWith,
    'mixWith': curriedMixWith,
}

TRANSFORM_KEYS = [
    'changeGain',
    'addCompression',
    'mixWith'
]


def main():
    target = FLAGS.output.split('/')
    for i, directory in enumerate(target):
        directory = '/'.join(target[0:i + 1])
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    files = readFolder(FLAGS.input)
    
    for file in files:
        for i in range(0, int(FLAGS.number_of_transforms)):
            transform = random.choice(TRANSFORM_KEYS)
            print(file, i, transform)
            transform_fn = TRANSFORMS[transform]
            sound = augmentSound('%s/%s' % (FLAGS.input, file), transform_fn)
            sound.export('%s/%d_%s_%s' % (FLAGS.output, i, transform, file))
                    
if __name__ == '__main__':
    main()