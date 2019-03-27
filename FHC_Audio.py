from pydub  import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import os
import wave
import librosa
from sklearn.cluster import KMeans
import logging
import pdb
import shutil

def helloFHC():
    print("Hello FHC!")


'''
    # return:
    # 0: self.getnchannels()：返回声道数，1标示单声道，2表示立体声 
    # 1: self.getsampwidth()：采样带宽，即每一帧的字节宽度。单位为byte。比如2表示2byte，16位。
    # 2: self.getframerate()：采样频率，比如48000表示48 kHz
    # 3: self.getnframes()：返回帧数。
    # 4: self.getcomptype():返回压缩类型（只支持 'NONE' 类型）
    # 5: self.getcompname():getcomptype() 的通俗版本。使用 'not compressed' 代替 'NONE'。
'''
### getting the basic information and draw the figures of the wave file
def basicInformationOfWavFile(FilePath):
    with wave.open(FilePath, 'r') as myWave: 
        params = myWave.getparams()
        print("The number of channels is {0}.\n \
        The sample-width is {1}.\n \
        The frame rate is {2}\n \
        The sampling frequency is {3}.\n \
        The total number of the frames is {4}.\n \
        ".format(params[0], params[1], params[2], params[3], params[4]))
        return params
    pass


### combining wave files
def combineWaveFiles(DIRpath):
    files = os.listdir(DIRpath)
    combinedSound = AudioSegment.empty()
    for f in files:
        if not os.path.isdir(f):
            tempfile = DIRpath + str(f)
            tempSound = AudioSegment.from_wav(tempfile)
            combinedSound += tempSound
            combinedSound.export(DIRpath + "_CombinedFile.wav", format="wav")  
    pass

### This method is dedicated to delete the silent chunks from wave file.
def stripSilenceFromWaveFile(FilePath, min_silence_len, silence_thresh):
    resultFolder = os.path.dirname(FilePath)
    print("The resultFolder is " + resultFolder)
    originalFile = AudioSegment.from_file(FilePath, format="wav")
    print("FilePath[len(resultFolder)+1:] is " + FilePath[len(resultFolder)+1:])
    chunks = split_on_silence(originalFile, min_silence_len, silence_thresh)
    newWaveFile = AudioSegment.empty()

    for c in chunks:
        newWaveFile = newWaveFile + c + AudioSegment.silent(duration=300)
    
    newWaveFile.export(resultFolder + '/' + "NonSilence_" + FilePath[len(resultFolder)+1:], format="wav")
    pass

def stripSilenceFromWaveFile(FilePath, resultFolder, min_silence_len, silence_thresh):
    print("The resultFolder is " + resultFolder)
    originalFile = AudioSegment.from_file(FilePath, format="wav")
    print("FilePath[len(resultFolder)+1:] is " + FilePath[len(resultFolder)+1:])
    chunks = split_on_silence(originalFile, min_silence_len, silence_thresh)
    newWaveFile = AudioSegment.empty()

    for c in chunks:
        newWaveFile = newWaveFile + c + AudioSegment.silent(duration=300)
    
    newWaveFile.export(resultFolder + '/' + "NonSilence_" + FilePath[len(resultFolder)+1:], format="wav")
    pass

def VolumeUpByAdd(FilePath, VolumeUpValue):
    resultFolder = os.path.dirname(FilePath)
    voice = AudioSegment.from_file(FilePath, format='wav') + VolumeUpValue
    voice.export(resultFolder + '/' + "VolumnUp_" + str(VolumeUpValue) + "dBs_" + FilePath[len(resultFolder)+1:], format="wav")
    pass

def VolumeDownByMinus(FilePath, VolumeDownValue):
    resultFolder = os.path.dirname(FilePath)
    voice = AudioSegment.from_file(FilePath, format='wav') + VolumeDownValue
    voice.export(resultFolder + '/' + "VolumnDown_" + str(VolumeDownValue) + "dBs_" + FilePath[len(resultFolder)+1:], format="wav")
    pass

def VolumeUpByMultiply(FilePath, VolumeUpTimes):
    resultFolder = os.path.dirname(FilePath)
    voice = AudioSegment.from_file(FilePath, format='wav') * VolumeUpTimes
    voice.export(resultFolder + '/' + "VolumnUp_" + str(VolumeUpTimes) + "times_" + FilePath[len(resultFolder)+1:], format="wav")
    pass

def VolumeDownByDivision(FilePath, VolumeDownTimes):
    resultFolder = os.path.dirname(FilePath)
    voice = AudioSegment.from_file(FilePath, format='wav') / VolumeDownTimes
    voice.export(resultFolder + '/' + "VolumnDown_" + str(VolumeDownTimes) + "times_" + FilePath[len(resultFolder)+1:], format="wav")
    pass


def findFilesContainCertainWords(RootDIR, text):
    fileList = []
    for parent, DIRs, files in os.walk(RootDIR):
        for f in files:
            if  f.find(text) != -1:
                file = os.path.abspath(os.path.join(parent,f))
                fileList.append(file)
    print(fileList)
    return fileList

def moveFiles(FileList, destinationDIR):
    for f in FileList:
        file = os.path.basename(f)
        print(f)
        if not os.path.exists(destinationDIR):
            os.mkdir(destinationDIR)
        shutil.copyfile(f, destinationDIR+'/'+file)
        print(f)




'''
This is a method for seperating audio files simply by 
min_silence_len: the length of minimal threshold for silence
and
silence_thresh: the minimal voice indicating silence.
AudioSegment lib is required.
'''
def naiveSeparateWaveFile(FilePath, min_silence_len=500, silence_thresh=-60):
    resultFolder = os.path.dirname(FilePath)
    originalFile = AudioSegment.from_file(FilePath, format="wav")
    chunks = split_on_silence(originalFile, min_silence_len, silence_thresh)

    for i, chunk in enumerate(chunks):
        chunk.export(resultFolder + '/' + FilePath[-5:] + "_" + str(i+1)+ "_SeparatedFile.wav", format="wav")
    pass

def getMFCC4AllFiles(FolderPath,n_mfcc):
    files = os.listdir(FolderPath)
    resultMFCC = np.zeros((1,n_mfcc))
    for file in files:
        if not os.path.isdir(file):
            targetFile = FolderPath + str(file)
            myWave = wave.open(targetFile, 'r')
            numFramerate = myWave.getframerate()
            sr = numFramerate
            myWave.close()
            y, sr= librosa.load(targetFile,sr=sr)
            originalMFCC = librosa.feature.mfcc(y=y, sr=numFramerate, n_mfcc=n_mfcc).T
            finalMFCC = np.mean(originalMFCC, axis = 0)/originalMFCC.shape[0]
            resultMFCC = np.vstack((resultMFCC,finalMFCC))
    resultMFCC = np.delete(resultMFCC,0,axis=0)
    return resultMFCC

def getMFCC4SingleFile(FilePath,n_mfcc):
    resultMFCC = np.zeros((1,n_mfcc))
    myWave = wave.open(FilePath, 'rb')
    sr = myWave.getframerate()
    myWave.close()
    #resultDIR = os.path.dirname(FilePath)
    y, sr= librosa.load(FilePath, sr=sr)
    resultMFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    return resultMFCC
    

def clusterFilesByKMeans(MFCCofFiles, numCluster):
    clf = KMeans(n_clusters=numCluster)
    clf.fit(MFCCofFiles)
    lablesOfMFCC = clf.labels_.reshape(-1,1)
    MFCCwithLabel = np.hstack((lablesOfMFCC, MFCCofFiles))
    return MFCCwithLabel



'''
###Extract the features from voice data
resultfolderPath = 'C:/'
sepratedFilesPath = 'c:/testData/seprate_wave/'
folderPath = 'c:/testData/combin_wave/'
targetFile = folderPath + 'output.wav'

myResult = getAllFeaturesOfAllFiles(sepratedFilesPath, 13)
MFCCwithLabel = clusterFilesByKMeans(myResult,2)

np.savetxt(resultfolderPath +'resultForKmeans.txt', myResult)
np.savetxt(resultfolderPath +'MFCCwithLabel.txt', MFCCwithLabel)


y, sr= librosa.load(targetFile,sr=44100)


print(sr)
print(y)

originalMFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
print("The shape of originalMFCC is {}".format(str(originalMFCC.shape)))

finalMFCC = np.mean(originalMFCC, axis = 0)/originalMFCC.shape[0]
print("The shape of finalMFCC is {}".format(str(finalMFCC.shape)))
print(finalMFCC)





###classify the wave files into two classes　by KMeans
clf = KMeans(n_clusters=2)
clf.fit(originalMFCC)
lablesOfMFCC = clf.labels_.reshape(-1,1)
print("The shape of lablesOfMFCC is {}".format(str(lablesOfMFCC.shape)))
MFCCwithLabel = np.hstack((lablesOfMFCC, originalMFCC))
print("The shape of MFCC is {}".format(str(originalMFCC.shape)))
print("The shape of MFCCwithLabel is {}".format(str(MFCCwithLabel.shape)))



np.save(resultfolderPath+'featureResult.npy', originalMFCC)
np.save(resultfolderPath+'forSVM.npy', MFCCwithLabel)

np.savetxt(resultfolderPath+'featureResult.txt', originalMFCC)
np.savetxt(resultfolderPath+'MFCCwithLabel.txt', MFCCwithLabel)
np.savetxt(resultfolderPath+'lables.txt', lablesOfMFCC)
'''
