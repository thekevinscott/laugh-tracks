#!apt-get install ffmpeg -y
#from pydub import AudioSegment
#from audioUtils import readFolderRecursive
#from audioInput import getOneHot, getFilePathsForClass, getSamplesAsVggishInput, calculateChunksForSamples, calculateMsForSamples, calculateMsForChunks
import random
import numpy as np
from audioInput import readFolderRecursive, calculateChunksForSamples, calculateChunksForMs, calculateMsForChunks, getSamplesAsVggishInput
from pydub import AudioSegment
from audio_transforms import mixWithFolder
from audio_transforms import changeGain, addCompression, changePitch
from tqdm import tqdm
import math

def getOneHot(class_num, idx):
    arr = np.zeros(class_num)
    arr[idx] = 1
    return arr

def getPosition(i):
    #mult = 16
    mult = 8
    return (calculateMsForChunks(i) + (i * mult))

cache = {}
def gatherData(files, transforms = [], start_at_zero = False):
    #print('im punting on this and just discarding chunks that dont have audio')
    chunks_of_audio = []
    #print('collecting files', files)
    print('gathering data for %i files' % (len(files)))
    
    for i in tqdm(range(0, len(files))):
        file = files[i]
        if file in cache:
            audio_segment = cache[file]
        else:
            #print('file', file)
            audio_segment = AudioSegment.from_file(file).set_channels(1)
            cache[file] = audio_segment
            
        starting_index = 0
        if start_at_zero:
            starting_index = random.randint(0, 960)

        sliced_audio = audio_segment[starting_index:]
        #print('length of sliced audio', len(sliced_audio))
        #print('expected chunks', calculateChunksForMs(len(sliced_audio)))

        audio = None
        if len(transforms) == 0:
            audio = sliced_audio
        else:
            #print('length of audio file', len(sliced_audio), file)
            for transform in transforms:
                #print('next transform is starting')
                transformed_audio = transform(sliced_audio)
                assert len(sliced_audio) == len(transformed_audio), "Length of transformed audio doesn't match, orig: %i, transformed: %i" % (len(sliced_audio), len(transformed_audio))
                if audio is None:
                    audio = transformed_audio
                else:
                    audio += transformed_audio
            #print('done, now get samples')
        samples = audio.get_array_of_samples()
        
        expected_chunks = calculateChunksForSamples(len(samples))
        #print('processing %i chunks' % (expected_chunks))
        #print('expected chunks based on samples', expected_chunks)
        for i in range(0, expected_chunks):
            start = getPosition(i)
            end = getPosition(i + 1)
            if start < len(audio):
                #print('start', start)
                assert start < len(audio), "Start time is greater than the length of the audio: start_time: %f, length of audio: %f" % (start, len(audio))
                chunk_of_audio = audio[start:end]
                chunks_of_audio.append({
                    'audio': chunk_of_audio,
                    'file': file,
                    'starting_index': starting_index + start,
                })
            
    return chunks_of_audio

def splitData(data, split):
    size = len(data)
    big = data[:round(size * (1 - split))]
    small = data[round(size * (1 - split)):]
    return big, small

"""Takes a number of dirs, gathers chunks for each file in that dir,
and then processes those chunks to return x and y
"""
def gatherTrainingDataWithCaching(dirs, augment_folders, should_balance = True, number_of_augmentations = 1):
    #print('gathering training data with caching')
    fileDirs = []
    augmentations = {}
    preprocessed_chunks = {}
    for i, d in enumerate(dirs):
        files = readFolderRecursive(d)
        fileDirs.append(files)
        
        augment_folder = augment_folders[i]
        if augment_folder:
            aug = mixWithFolder(augment_folder, [
                    {'name': 'gain', 'fn': lambda audio: changeGain(audio, 20, 10) },
                    {'name': 'compression', 'fn': lambda audio: addCompression(audio) },
                    # this messes the timing of the sample when running through vggish.
                    # need to find a method that doesn't change the audio length
                    #{'name': 'pitch', 'fn': lambda audio: changePitch(audio, -0.3, max=0.3) },
                ])
            augs = []
            for _ in range(number_of_augmentations):
                augs.append(aug)
            augmentations[i] = augs
        else:
            augmentations[i] = None
            label = getOneHot(len(dirs), i)
            chunks = gatherData(files, transforms = [])
            for chunk in chunks:
                chunk['label'] = label
            preprocessed_chunks[i] = chunks

        
    def curriedFn(split = 0.2, shuf = True):
        #print('gathering training data')
        chunks_of_audio = []
        for i, files in enumerate(fileDirs):
            transforms = augmentations[i]
            
            if transforms:
                print('gathering training data for on the fly with transforms', files)
                chunks = gatherData(files, transforms = transforms)
                label = getOneHot(len(dirs), i)
                for chunk in chunks:
                    chunk['label'] = label
            else:
                print('these files have already been processed', files)
                chunks = preprocessed_chunks[i]

            chunks_of_audio += chunks
            
        
        if should_balance:
            chunks_of_audio = balanceData(chunks_of_audio)
            
           
        if shuf:
            #print('shuffle the files')
            random.shuffle(chunks_of_audio)
        #print('now we split the files')
        train, test = splitData(chunks_of_audio, split)
        #return train
        return preprocessForTraining(train), preprocessForTraining(test)
    return curriedFn

def fillUp(arr, m):
    times = math.floor(m/len(arr))
    remainder = m - (times * len(arr))
    return arr * times + arr[0:remainder]

def balanceData(chunks):
    chunks_by_label = {}
    for chunk in chunks:
        label = chunk['label'].tostring()
        #print('label', label)
        if label not in chunks_by_label:
            chunks_by_label[label] = []
            
        chunks_by_label[label].append(chunk)
        #print('chunk', chunk)
        
    amount = 0
    #print(chunks_by_label)
    
    for key in chunks_by_label.keys():
        if len(chunks_by_label[key]) > amount:
            amount = len(chunks_by_label[key])
    #print('the max amount', amount)    
    chunks = []
    for key in chunks_by_label.keys():
        #print('giving it in', len(chunks_by_label[key]))
        filled_up_chunks = fillUp(chunks_by_label[key], amount)
        #print('filled up chunks', len(filled_up_chunks))
        chunks += filled_up_chunks
    #print('chunks', chunks)       
        
    return chunks
    
def gatherTrainingData(dirs, split = .2, shuf = True, augment_folders = None):
    print('gathering training data')
    chunks_of_audio = []
    for i, d in enumerate(dirs):
        files = readFolderRecursive(d)
        print('gathering training data for', files)        
        transforms = []
        augment_folder = augment_folders[i]

        if augment_folder is not '' and augment_folder is not None:
            transforms = [
                #lambda audio: audio,
                #lambda audio: changePitch(audio, min = -0.5, max = 0.5),
                #lambda audio: changeGain(audio),
                #lambda audio: addCompression(audio),
                mixWithFolder(augment_folder, [
                    lambda audio: changeGain(audio, 20, 10),
                    lambda audio: changePitch(audio, -0.3, max=0.3),
                    lambda audio: addCompression(audio),
                ]),
            ]
        chunks = gatherData(files, transforms = transforms)
        
        
        label = getOneHot(len(dirs), i)
        for chunk in chunks:
            chunk['label'] = label
            
        chunks_of_audio += chunks
    if shuf:
        random.shuffle(chunks_of_audio)
    train, test = splitData(chunks_of_audio, split)
    #return train
    return preprocessForTraining(train), preprocessForTraining(test)

def gatherTestingData(files):
    chunks_of_audio = gatherData(files, transforms = [], start_at_zero = True)
    return preprocessForTesting(chunks_of_audio) 

def concatSamples(chunks):
    all_samples = None
    for chunk in chunks:
        #print('chunk', chunk)
        if 'samples' in chunk:
            
            samples = chunk['samples']
            if all_samples is None:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples,samples), axis=0)
    return all_samples

def preprocessForTesting(chunks):
    for i, chunk in enumerate(chunks):
        audio = chunk['audio']
        #print('chunk', chunk['file'])
        samples = audio.get_array_of_samples()
        #a = np.array(samples)
        #print(np.max(a), np.min(a), np.mean(a))
        expected_chunks = calculateChunksForSamples(len(samples))
        assert expected_chunks <= 1, "Expected chunks is greater than 1, something is dearly wrong, %i, %s " % (expected_chunks, chunk['file'])
        if expected_chunks > 0:
            
            #assert len(samples) > 0, "Audio is empty: %s, starting index: %s, enumerated index: %i " % (chunk['file'], chunk['starting_index'], i)
            #print('len of samples', len(samples))
            #print(chunk)
            vggish_samples = getSamplesAsVggishInput(samples)

            #a = np.array(vggish_samples)
            #print(np.max(a), np.min(a), np.mean(a), vggish_samples[0][0][0])
            chunk['samples'] = vggish_samples
    #return chunks

    #print('first chunk')
    #print(chunks[0]['samples'])
    
    return concatSamples(chunks), chunks


def preprocessForTraining(chunks):
    all_samples = None
    labels = []
    print('preprocess for training', len(chunks))

    for i in tqdm(range(0, len(chunks))):
        chunk = chunks[i]
        audio = chunk['audio']
        
        samples = audio.get_array_of_samples()
        expected_chunks = calculateChunksForSamples(len(samples))
        expected_chunks_for_ms = calculateChunksForMs(len(audio))


        #print(chunk)
        #print('length of audio', len(audio))
        #print('length of samples', len(samples))
        #print('expected chunks for ms', expected_chunks_for_ms)
        assert expected_chunks <= 1, "Expected chunks is %i and should be 1 or less, something is dearly wrong, %s " % (expected_chunks, chunk['file'])

        if expected_chunks > 0:
            #assert len(samples) > 0, "Audio is empty: %s, starting index: %s, enumerated index: %i " % (chunk['file'], chunk['starting_index'], i)
            #print('len of samples', len(samples))
            print('chunk', chunk)
            vggish_samples = getSamplesAsVggishInput(samples)

            #print('vggish samples', len(vggish_samples))
            assert len(vggish_samples) == 1, "VGGish returned not 1, but %i: %s, starting index: %s, enumerated index: %i " % (len(vggish_samples), chunk['file'], chunk['starting_index'], i)
            chunk['samples'] = vggish_samples
    
    for chunk in chunks:
        if 'samples' in chunk:
            labels.append(chunk['label'])
    all_samples = concatSamples(chunks)
            
    assert np.array(all_samples).shape[1] == 96, "Check shape of final samples, %s" % (np.array(all_samples).shape)
    assert len(labels) == len(all_samples), "Sizes don't match post vggish calc, labels: %i, samples: %i" % (len(labels), len(all_samples))
    return all_samples, labels