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
from audioInput import getLaughTracks, getNoise, processWavFile
import IPython.display as ipd
from IPython.core.display import display, HTML
import random
from audioUtils import readFolder
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import random
import json

def displayAudio(path):
    display(ipd.Audio(path))

def displayAudioForFolder(path):
    files = readFolder(path)
    for file in files:
        full_file = '%s/%s' % (path, file)
        displayAudio(full_file)
    
def printResults(preds): 
    with tf.Graph().as_default(), tf.Session() as sess:
        return sess.run(tf.argmax(input=preds, axis=1))

def predictAudio(path, model_name, label_translations, number_of_classes = 2, multiplier = 1):
    model = getModel(model_name)
    wavfile = processWavFile(path, log=False)
    preds = predict(model, number_of_classes, wavfile)
    prettyPreds = printResults(preds)
    translatedPreds = []    
    for p in prettyPreds:
        translatedPreds.append(label_translations[p])
    displayWaveform(path, translatedPreds, multiplier = multiplier)
    return prettyPreds, wavfile

def getModel(path):
    files = readFolder('model/%s' % path)
    if len(files) > 0:
        return '%s/%s' % (path, files[0])
    return None
    
def displayWaveform(path, preds, multiplier = 1):
    id = 'i%i' % random.randint(1,10000001)    
        
    preds = json.dumps(preds)
    script = '''
    <script src="https://thekevinscott.github.io/laugh-tracks/frontend-dist/static/js/main.8e06b391.js"></script>
    <script type="text/javascript">
    (function() {
        var jsonPreds = %s;
        var predMarkers = [];
        function getTimestamp(i) {
            i += 1
            return (976*i-(16*(i-1)) / 1000 - 976) / 1000;
        }
        jsonPreds.forEach(function(pred, i) {
            const timestamp = getTimestamp(i) * %f;
            console.log(i, timestamp);
            
            predMarkers.push({
                label: pred,
                timestamp,
            })
        })
        waveform.default(document.getElementById('%s'), '%s', predMarkers);
        
    })()
    </script>
    ''' % (preds, multiplier, id, path)
    style = '''
    <style type="text/css">
 #waveform {
 width: 200px;
 height: 200px;
 background: #EEE;
 }
    </style>
    '''
    args = {'id':id, 'path':path, 'script': script, 'style': style}
        
    html = '''
    {style}
    <div id="{id}"></div>
    {script}
'''.format(**args)
    display(HTML(html)) 