#from audioModel import predict, train, accuracy, getCorrectAndIncorrect
#from audioInput import getLaughTracks
import tensorflow as tf
from audioModel import predict
from audioInput import readFolderRecursive
from audioDisplay import getModel
from audioInput_v2 import gatherTestingData
from audioDisplay import printResults

flags = tf.app.flags

flags.DEFINE_string('files', None, 'Files for prediction, comma separated')
flags.DEFINE_string('dirs', None, 'Directories containing files for prediction, comma separated')
flags.DEFINE_string('classes', None, 'Number of classes to expect, comma separated')
flags.DEFINE_string('model_name', None, 'The model name to load')

FLAGS = flags.FLAGS

assert FLAGS.model_name is not None, "Provide a model name"
assert FLAGS.classes is not None, "Provide classes as comma separated strings"


if FLAGS.dirs:
    dirs = FLAGS.dirs.split(',')
    files = []
    for d in dirs:
        files += readFolderRecursive(d)  
else:
    files = FLAGS.files.split(',')

x, chunks = gatherTestingData(files)
model_name = getModel(FLAGS.model_name)
classes = FLAGS.classes.split(',')

print('using model', model_name)
print('predicting files', files)

preds = predict(model_name, len(classes), x)
prettyPreds = printResults(preds)
translatedPreds = []    
for p in prettyPreds:
    translatedPreds.append(classes[p])

print('translated preds', translatedPreds)