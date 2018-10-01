#from audioModel import predict, train, accuracy, getCorrectAndIncorrect
#from audioInput import getLaughTracks
import tensorflow as tf
from audioInput_v2 import gatherTrainingDataWithCaching, preprocessForTraining
from audioModel import train, predict

import vggish_params

flags = tf.app.flags

flags.DEFINE_string('dirs', '', 'Dirs to use for training, comma separated')
flags.DEFINE_string('epochs', '1', 'Number of epochs')
flags.DEFINE_string('batch_size', '32', 'The batch size')
flags.DEFINE_string('model_name', 'audio', 'The model name to save')
flags.DEFINE_string('augment_folders', '', 'The folder to use to augment the sample data')
flags.DEFINE_string('should_balance', '1', 'Whether to balance the incoming datasets')
flags.DEFINE_string('number_of_augmentations', '1', 'The number of augmentations (mixing with noise) to perform')


#flags.DEFINE_string('mix_dir', None, 'The directory to include for mixing files')
#flags.DEFINE_string('number_of_transforms', '5', 'Number of times to transform file')
FLAGS = flags.FLAGS

    
number_of_samples = None
lr=vggish_params.LEARNING_RATE
dirs = FLAGS.dirs.split(',')
augment_folders = FLAGS.augment_folders.split(',')

curriedGetData = gatherTrainingDataWithCaching(dirs, 
                                               augment_folders = augment_folders, 
                                               should_balance = FLAGS.should_balance == '1', 
                                               number_of_augmentations = int(FLAGS.number_of_augmentations))

train(lambda shuf : curriedGetData(split = .1, shuf=shuf), 
      len(dirs), 
      model_name = FLAGS.model_name, 
      epochs = int(FLAGS.epochs), 
      batch_size = int(FLAGS.batch_size))

#(features, labels, chunks) = getSamples(['laughter-test', 'notlaughter-test'], shuf = False, number_of_samples = None, log=False)
#preds = predict(getModel('%s' % (model_name)), number_of_classes, test_data)
#printResults(preds, labels)