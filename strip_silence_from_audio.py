from __future__ import print_function

from audioUtils import stripSilenceInPlace
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('file', None, 'the file to strip')

flags.DEFINE_string(
    'threshold_beneath_average', '30', 'i dunno what this does')


FLAGS = flags.FLAGS

def main():
    stripSilenceInPlace(FLAGS.file, FLAGS.threshold_beneath_average)
                    
if __name__ == '__main__':
    main()