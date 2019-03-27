from pydub  import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import os
import wave
import librosa
from sklearn.cluster import KMeans


print("Test starts!")


### Obtaining the basic information of wave file. 
### It is not necessary if you do not need to get the information of voice data.
        # 0: self.getnchannels()：返回声道数，1标示单声道，2表示立体声 
        # 1: self.getsampwidth()：采样带宽,即每一帧的字节宽度。单位为byte。比如2表示2byte，16位。
        # 2: self.getframerate()：采样频率,比如48000表示48 kHz
        # 3: self.getnframes()：返回帧数。
        # 4: self.getcomptype():返回压缩类型（只支持 'NONE' 类型）
        # 5: self.getcompname():getcomptype() 的通俗版本。使用 'not compressed' 代替 'NONE'。
'''
folderPath = 'c:/testData/combin_wave/'
targetFile = folderPath + 'output.wav'

with wave.open(targetFile, 'r') as myWave:
    
    # return:
    # 0: self.getnchannels()：返回声道数，1标示单声道，2表示立体声 
    # 1: self.getsampwidth()：采样带宽，即每一帧的字节宽度。单位为byte。比如2表示2byte，16位。
    # 2: self.getframerate()：采样频率，比如48000表示48 kHz
    # 3: self.getnframes()：返回帧数。
    # 4: self.getcomptype():返回压缩类型（只支持 'NONE' 类型）
    # 5: self.getcompname():getcomptype() 的通俗版本。使用 'not compressed' 代替 'NONE'。
    for item in enumerate(myWave.getparams()):
        print(item

  
    num_frame = myWave.getnframes()
    num_Channel = myWave.getnchannels()
    num_framerate = myWave.getframerate()

    #readframes():从流的当前指针位置一次读出音频的n个帧，并且指针后移n个帧，返回一个字节数组。
    #readframes返回的是二进制数据,在Python中用字符串表示二进制数据.
    voiceStrData = myWave.readframes(num_frame)

voiceNumData = np.fromstring(voiceStrData, dtype=np.short)
voiceNumData.shape = -1,num_Channel
#voiceNumData = voiceNumData.T

###########################################################################
'''

'''
### combining wave files
folderPath = 'c:/testData/combin_wave/'
files = os.listdir(folderPath)


combinedSound = AudioSegment.from_wav(folderPath+"1.wav")
for file in files:
    if not os.path.isdir(file):
        tempfile = folderPath + str(file)
        tempSound = AudioSegment.from_wav(tempfile)
        combinedSound = combinedSound + tempSound
        combinedSound.export(folderPath + "output.wav", format="wav")    
###########################################################################
'''

'''
### seprate the voice into slice
folderPath = 'c:/testData/combin_wave/'
resultfolderPath = 'c:/testData/seprate_wave/'
targetFile = folderPath + 'output.wav'

originalFile = AudioSegment.from_file(targetFile, format="wav")
chunks = split_on_silence(originalFile, min_silence_len=300, silence_thresh=-55)

for i, chunk in enumerate(chunks):
    chunk.export(resultfolderPath+targetFile[0:5]+'_'+str(i+1)+'.wav', format='wav')
###########################################################################
'''



def getMFCC4AllFiles(folderPath,n_mfcc):
    files = os.listdir(folderPath)
    resultMFCC = np.zeros((1,n_mfcc))
    #################################################################################
    # get MFCC for each file
    for file in files:
        if not os.path.isdir(file):
            targetFile = folderPath + str(file)
            myWave = wave.open(targetFile, 'r')
            numFramerate = myWave.getframerate()
            sr = numFramerate
            myWave.close()
            y, sr= librosa.load(targetFile,sr=sr)
            originalMFCC = librosa.feature.mfcc(y=y, sr=numFramerate, n_mfcc=n_mfcc).T
            finalMFCC = np.mean(originalMFCC, axis = 0)/originalMFCC.shape[0]
            resultMFCC = np.vstack((resultMFCC,finalMFCC))
    ###################################################################################
    resultMFCC = np.delete(resultMFCC,0,axis=0)
    return resultMFCC

def get4AllFiles(folderPath,n_mfcc):
    files = os.listdir(folderPath)
    resultMFCC = np.zeros((1,n_mfcc))
    #################################################################################
    # get MFCC for each file
    for file in files:
        if not os.path.isdir(file):
            targetFile = folderPath + str(file)
            myWave = wave.open(targetFile, 'r')
            numFramerate = myWave.getframerate()
            sr = numFramerate
            myWave.close()
            y, sr= librosa.load(targetFile,sr=sr)
            originalMFCC = librosa.feature.mfcc(y=y, sr=numFramerate, n_mfcc=n_mfcc).T
            finalMFCC = np.mean(originalMFCC, axis = 0)/originalMFCC.shape[0]
            resultMFCC = np.vstack((resultMFCC,finalMFCC))
    ###################################################################################
    resultMFCC = np.delete(resultMFCC,0,axis=0)
    return resultMFCC

def clusterFilesByKMeans(MFCCofFiles, numCluster):
    clf = KMeans(n_clusters=numCluster)
    clf.fit(MFCCofFiles)
    lablesOfMFCC = clf.labels_.reshape(-1,1)
    MFCCwithLabel = np.hstack((lablesOfMFCC, MFCCofFiles))
    return MFCCwithLabel

###Extract the features from voice data
resultfolderPath = 'C:/'
sepratedFilesPath = 'c:/testData/seprate_wave/'
folderPath = 'c:/testData/combin_wave/'
targetFile = folderPath + 'output.wav'

myResult = getAllFeaturesOfAllFiles(sepratedFilesPath, 13)
MFCCwithLabel = clusterFilesByKMeans(myResult,2)

np.savetxt(resultfolderPath +'resultForKmeans.txt', myResult)
np.savetxt(resultfolderPath +'MFCCwithLabel.txt', MFCCwithLabel)


print('ok')
'''
y, sr= librosa.load(targetFile,sr=44100)


print(sr)
print(y)

originalMFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
print("The shape of originalMFCC is {}".format(str(originalMFCC.shape)))

finalMFCC = np.mean(originalMFCC, axis = 0)/originalMFCC.shape[0]
print("The shape of finalMFCC is {}".format(str(finalMFCC.shape)))
print(finalMFCC)
'''
###########################################################################



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

###########################################################################



print("Test ends!")