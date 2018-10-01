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
from audioUtils import readFolder, mkdir

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
        max = 15
        overlay_file = random.choice(noises)
        noise = AudioSegment.from_file(overlay_file)
        gained_noise = changeGain(noise, max, int(max / 2))
        if len(gained_noise) < len(sound):
            orig_gained_noise = gained_noise
            #print('extend the noise over the main sound')
            times = len(sound)/len(gained_noise)
            for i in range(1, math.ceil(times)):
                gained_noise = gained_noise + orig_gained_noise
            
        
        #print('original', len(sound), 'noise', len(gained_noise))
        # lets say the original is 100, and the overlay is 200
        # you'd pick a random number from 0-100
        length = len(sound)
        rand_start = random.randint(0, len(gained_noise) - length)
        return changeGain(sound, max, int(max / 2)).overlay(gained_noise[rand_start:rand_start + length])
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
}

TRANSFORM_KEYS = [
    'changeGain',
    'addCompression',
    'mixWith',
    'mixWith',
    'mixWith',
    'mixWith',
    'mixWith',
    'mixWith',
]


def main():
    #audio = augmentSound('samples/laughter/standup 1 laughter.wav', mixWith([
    #    'noise_samples/8c5YY9DcoiE.wav',
    #]))
    #audio.export('tmp.wav')
    #return
    files = readFolder(FLAGS.input)
    mkdir(FLAGS.output)
    for file in files:
        sound = AudioSegment.from_file('%s/%s' % (FLAGS.input, file))
        
        file_parts = file.split('.')
        file_name = '.'.join(file_parts[0:len(file_parts) - 1])
        #exported_file = '%s/%s' % (FLAGS.output, file_name)
        #sound.export(exported_file)
        for i in range(0, int(FLAGS.number_of_transforms)):
            #transform = TRANSFORM_KEYS[i]
            transform = random.choice(TRANSFORM_KEYS)
            print(file, i, transform)
            transform_fn = TRANSFORMS[transform]
            sound = augmentSound('%s/%s' % (FLAGS.input, file), transform_fn)
            exported_file = '%s/%s_%d_%s.wav' % (FLAGS.output, file_name, i + 1, transform)
            sound.export(exported_file)
                    
if __name__ == '__main__':
    main()