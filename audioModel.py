from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf
import os
import vggish_input
import vggish_params
import vggish_slim
from pydub import AudioSegment
from audioUtils import readFolder, shell

slim = tf.contrib.slim

def loadVGGish(sess, number_of_classes):
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
            learning_rate=vggish_params.LEARNING_RATE,
            epsilon=vggish_params.ADAM_EPSILON)
        optimizer.minimize(loss, global_step=global_step, name='train_op')

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, './vggish_model.ckpt') 
    return logits, pred

def saveModel(sess, model_name):
    model_folder = './model/%s' % model_name
    model_name_to_save = '%s/model' % (model_folder)    
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    saver = tf.train.Saver()
    saver.save(sess, model_name_to_save)
    
def deleteModel(model_name):
    model_folder = './model/%s' % model_name
    shell('rm -rf %s' % model_folder)
    
def train(get_examples, number_of_classes, model_name='foo', epochs = 50):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        logits, pred = loadVGGish(sess, number_of_classes)

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
            (features, labels) = get_examples(shuf=True)
            [num_steps, loss, _] = sess.run(
                [global_step_tensor, loss_tensor, train_op],
                feed_dict={features_tensor: features, labels_tensor: labels})
            print('Step %d: loss %g' % (num_steps, loss))
            
            model_id = '%s_%s-%s' % (model_name, epoch + 1, epochs)
            saveModel(sess, model_id)
            deleteModel('%s_%s-%s' % (model_name, epoch, epochs))

def predict(model_name, number_of_classes, features):
    print('number of classes', number_of_classes)
    model_name_to_load = './model/%s/model' % (model_name)   
    print('loading', model_name_to_load)
    #model_name_to_load = './model/%s' % (model_name)   
    
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