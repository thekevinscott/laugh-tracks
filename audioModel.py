from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf
import os
import vggish_input
from tqdm import tqdm
import vggish_params
import vggish_slim
from pydub import AudioSegment
from audioUtils import readFolder, shell
from audioInput import getSamplesAsVggishInput, shuffleSamples


slim = tf.contrib.slim

def loadVGGish(sess, number_of_classes, lr = vggish_params.LEARNING_RATE):
    embeddings = vggish_slim.define_vggish_slim(True) # Do we train VGG-ish?

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
        # Add a fully connected layer with 100 units.
        num_units = 100
        fc = slim.fully_connected(embeddings, num_units)

        # Add a classifier layer at the end, consisting of parallel logistic
        # classifiers, one per class. This allows for multi-class tasks.
        logits = slim.fully_connected(
          fc, number_of_classes, activation_fn=None, scope='logits')
        pred = tf.sigmoid(logits, name='prediction')

        # Add training ops.
        with tf.variable_scope('train'):
            global_step = tf.Variable(
                0, name='global_step', trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                             tf.GraphKeys.GLOBAL_STEP])

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels = tf.placeholder(
            tf.float32, shape=(None, number_of_classes), name='labels')

        # Cross-entropy label loss.
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xent')
        loss = tf.reduce_mean(xent, name='loss_op')
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            epsilon=vggish_params.ADAM_EPSILON)
        optimizer.minimize(loss, global_step=global_step, name='train_op')

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, './vggish_model.ckpt') 
    return logits, pred

def saveModel(sess, model_name, model_id):
    model_folder = './model/%s' % model_name
    model_folder_id = '%s/%s' % (model_folder, model_id)    
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    if not os.path.isdir(model_folder_id):
        os.mkdir(model_folder_id)        
    saver = tf.train.Saver()
    path = '%s/model' % model_folder_id
    print('saving model', path)
    saver.save(sess, path)
    
def deleteModel(model_name, model_id):
    path = './model/%s/%s' % (model_name, model_id)
    if os.path.isdir(path):    
        #print('deleting model', path)
        shell('rm -rf %s' % path)

def getCorrectAndIncorrect(preds, labels):
    correct = []
    incorrect = []
    max_preds = np.argmax(preds, 1)
    max_labels = np.argmax(labels, 1)
    for i, pred in enumerate(max_preds):
        distance = 0
        for j, _ in enumerate(preds[i]):
            distance += abs(preds[i][j] - labels[i][j])
        obj = {
            'i': i,
            'pred': preds[i],
            'label': labels[i],
            'distance': distance,
        }
        if pred == max_labels[i]:
            correct.append(obj)
        else:
            incorrect.append(obj)
    correct = sorted(incorrect, key=lambda p: p['distance'], reverse=False)
    incorrect = sorted(incorrect, key=lambda p: p['distance'], reverse=True)
    return correct, incorrect        

def train_old(get_examples, number_of_classes, model_name='foo', epochs = 50, batch_size = 32, lr = vggish_params.LEARNING_RATE):
    VALIDATION_SPLIT = 0.1    
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        logits, pred = loadVGGish(sess, number_of_classes, lr=lr)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        #for op in tf.get_default_graph().get_operations():
            #print(str(op.name))

        labels_tensor = sess.graph.get_tensor_by_name('mymodel/labels:0')
        #labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')    
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/loss_op:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train_op')

        # The training loop.
        for epoch in range(epochs):
            print('get examples', get_examples)
            (features, labels, chunks) = get_examples(shuf=True)

            if len(features) * VALIDATION_SPLIT < 1:
                print('you might have issues since your shape is less than the validation split')
            
            train_x = features[0:round(len(features)*(1-VALIDATION_SPLIT))]
            train_y = labels[0:round(len(features)*(1-VALIDATION_SPLIT))]
            validation_x = features[round(len(features)*(1-VALIDATION_SPLIT)):]
            validation_y = labels[round(len(features)*(1-VALIDATION_SPLIT)):]
            
            for i in tqdm(range(0, len(train_x), batch_size)):
                X_train_mini = train_x[i:i + batch_size]
                y_train_mini = train_y[i:i + batch_size]

                [num_steps, loss, _] = sess.run(
                    [global_step_tensor, loss_tensor, train_op],
                    feed_dict={features_tensor: X_train_mini, labels_tensor: y_train_mini})
                
                assert not math.isnan(loss), "Loss is nan"

            preds = sess.run(pred, feed_dict={features_tensor: validation_x})
            acc = accuracy(preds, validation_y)
            print('Epoch %d: loss %g, acc %f' % (epoch, loss, acc))
            saveModel(sess, model_name, '%s_%s' % (epoch + 1, epochs))
            deleteModel(model_name, '%s_%s' % (epoch, epochs))    
            
            
            
def train(get_examples, number_of_classes, model_name='foo', epochs = 50, batch_size = 32, lr = vggish_params.LEARNING_RATE):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        logits, pred = loadVGGish(sess, number_of_classes, lr=lr)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        #for op in tf.get_default_graph().get_operations():
            #print(str(op.name))

        labels_tensor = sess.graph.get_tensor_by_name('mymodel/labels:0')
        #labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')    
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/loss_op:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train_op')

        train, validation = get_examples(shuf=True)

        train_x, train_y = train
        validation_x, validation_y = validation
        
        # The training loop.
        for epoch in range(epochs):
            for i in tqdm(range(0, len(train_x), batch_size)):
                X_train_mini = train_x[i:i + batch_size]
                y_train_mini = train_y[i:i + batch_size]
                #print(epoch, i, np.array(X_train_mini).shape, np.array(y_train_mini).shape)
                [num_steps, loss, _] = sess.run(
                    [global_step_tensor, loss_tensor, train_op],
                    feed_dict={features_tensor: X_train_mini, labels_tensor: y_train_mini})

            preds = sess.run(pred, feed_dict={features_tensor: validation_x})
            acc = accuracy(preds, validation_y)
            print('Epoch %d: loss %g, acc %f' % (epoch, loss, acc))
            saveModel(sess, model_name, '%s_%s' % (epoch + 1, epochs))
            deleteModel(model_name, '%s_%s' % (epoch, epochs))

def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    sum = 0
    for i in range(len(pred_class)):
        if pred_class[i] == true_class[i]:
            sum += 1
    
    return sum / predictions.shape[0]
    
    
def predict(model_name, number_of_classes, features):
    #print('number of classes', number_of_classes)
    model_name_to_load = './model/%s/model' % (model_name)   
    #print('loading', model_name_to_load)
    #model_name_to_load = './model/%s' % (model_name)   
    #print("incoming features", features)
    with tf.Graph().as_default(), tf.Session() as sess:
        logits, pred = loadVGGish(sess, number_of_classes)
        saver = tf.train.Saver()        
        saver.restore(sess, model_name_to_load)  
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        prediction=tf.argmax(logits,1)
        embedding_batch = sess.run(pred, feed_dict={features_tensor: features})
        return embedding_batch 