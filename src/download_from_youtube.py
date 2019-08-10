from __future__ import print_function
import os
from audioUtils import stripSilenceInPlace
import tensorflow as tf
from audioUtils import readFolder, mkdir, downloadYtAndPrepareAudio

flags = tf.app.flags

flags.DEFINE_string('strip_silence', '1', 'whether to strip silence or not')
flags.DEFINE_string('target', None, 'directory to save to')
flags.DEFINE_string('ids', None, 'comma separated list of ids to download')


FLAGS = flags.FLAGS

def main():
    mkdir(FLAGS.target)
    for idx in FLAGS.ids.split(','):
        target = '%s/%s' % (FLAGS.target, idx)
        print('target', target)
        if not os.path.isfile('%s.wav' % target):
            directory = downloadYtAndPrepareAudio(idx, target, FLAGS.strip_silence == '1')
                    
if __name__ == '__main__':
    main()