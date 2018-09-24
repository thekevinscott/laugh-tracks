"""Strip silence from a wav file"""

from pydub import AudioSegment
from pydub.silence import split_on_silence
import subprocess
import numpy as np
import math
import os
#import IPython.display as ipd

def shell(cmd):
    subprocess.call(cmd, shell=True)

def stripSilenceFromAudio(file, threshold_beneath_average = 30):
    sound = AudioSegment.from_file(file)
    splitsound = split_on_silence(sound, silence_thresh=(sound.dBFS - threshold_beneath_average))
    combined = AudioSegment.empty()
    for split in splitsound:
        combined += split

    return combined

def getVid(id, target):
    youtube_path = 'http://www.youtube.com/watch?v=' + id
    shell('youtube-dl --extract-audio --audio-format wav --output %s %s' % (target, youtube_path)) 
    return target


def clearFolder(folder):
    if os.path.isdir(folder):
        shell('rm -rf %s' % (folder))
    os.mkdir(folder)

def readFolder(folder):
    pipe = subprocess.Popen('ls ' + folder, shell=True, stdout=subprocess.PIPE)
    files = []
    for line in pipe.stdout:
        files.append(line.strip().decode('ascii'))
    
    return files

def writeAudio(files, folder):
    if not isinstance(files, (list, tuple)):
        files = [files]

    for file in files:
        path = folder + '/' + file
        #print(path)
        display(Audio(path))

def readAndWriteAudio(folder):
    files = readFolder(folder)
    writeAudio(files, folder)

def downloadYtAndPrepareAudio(id, target, stripSilence, dtype = 'int16'):
    print('id', id)
    folder = '%s/' % (target)
    downloaded_yt = '%s%s_yt.wav' % (folder, id)
    prepared = '%s%s.wav' % (folder, id)
    #print(downloaded_yt)
    #print(prepared)

    #output_path_for_slices = '%s/out' % (folder)
    
    
    #clearFolder(output_path_for_slices)    
    
    getVid(id, downloaded_yt)
    if stripSilence:
        file = stripSilenceFromAudio(downloaded_yt)
        
    else:
        file = AudioSegment.from_file(downloaded_yt)
    
    file.export(prepared, format="wav", bitrate="44.1k")   
    shell('rm %s' % downloaded_yt)
    return prepared
    
    
    
    
    
# OLD IMPLEMENTATION
def downloadYtAndSliceBy(id, target, stripSilence, seconds_per_segment = 1, dtype = 'int16'):
    print('id', id, 'seconds per segment', seconds_per_segment)
    folder = '%s/%s' % (target, id)
    downloaded_yt = '%s/yt.wav' % (folder)
    stripped_file = '%s/stripped.wav' % (folder)    
    output_path_for_slices = '%s/out' % (folder)
    
    clearFolder(folder)
    clearFolder(output_path_for_slices)    
    
    getVid(id, downloaded_yt)
    if stripSilence:
        file = stripSilenceFromAudio(downloaded_yt)
        file.export(stripped_file, format="wav", bitrate="44.1k")
    else:
        file = AudioSegment.from_file(downloaded_yt)
        
    length = math.ceil(len(file) / 1000 / seconds_per_segment)
    for i in range(0, length):
        slice = sliceAudioFile(file, output_path_for_slices, seconds_per_segment, i, dtype)
    #return slice
    return output_path_for_slices


def sliceAudioFile(file, target, length = 1, start = 0, dtype = 'int16'):
    #file = AudioSegment.from_file(original)
    starting_ms = start * 1000 * length
    duration_ms = length * 1000
    ending_ms = starting_ms + duration_ms
    slice = file[starting_ms:ending_ms]
    samples = np.array(slice.get_array_of_samples()).astype('int16')
    target_filename = "%s/%f_%f.wav" %(target,starting_ms/1000,ending_ms/1000)

    audio_segment = AudioSegment(
        samples.tobytes(), 
        frame_rate=slice.frame_rate,
        sample_width=samples.dtype.itemsize, 
        channels=1
    )

    audio_segment.export(target_filename, format="wav", bitrate="44.1k")
    return slice