import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np

import FHC_Audio


targetDIR = "F:\Riku_Ka\Training_All\Training Data Set"
with open("F:/Riku_Ka/Training_All/files_for_Training_Data_Set.txt", 'w') as targetFile:
    files = os.listdir(targetDIR)
    for f in files:
        targetFile.write(f[0:11] + "\n")
        

'''
sourceDIR = "F:/Riku_Ka/PROMPT-UnzipAudioFile/音声データ"
dstDIR = "F:/Riku_Ka/Training_All/MFCC for Depression-Pt only"
subDIRs = os.listdir(sourceDIR)
#listHealth = ['BH', 'KH', 'OH', 'PKH', 'POH', 'PTH', 'TH']
listDepression = ['AD', 'BD', 'GD', 'KD', 'MD', 'ND', 'OD', 'PKD', 'POD', 'PTD', 'SD', 'TD']
print(listDepression)

for dirs in subDIRs:
    tempDIR = sourceDIR + '/' + dirs
    files = os.listdir(tempDIR)
    for f in files:
        if any(x in f for x in listDepression) and f.find('Pt') != -1:
            tempFILE = tempDIR + '/' + f
            featureMFCC = FHC_Audio.getMFCC4SingleFile(tempFILE, 13)
            featureFILE = dstDIR + '/' + os.path.splitext(f)[0] +"_MFCC.txt"
            np.savetxt(featureFILE, featureMFCC)
'''            

'''            
files = os.listdir(trainingDIR)
print(files)
for f in files:
    print(f)
    tempFile = trainingDIR + '/' + f
    resultMFCC =  FHC_Audio.getMFCC4SingleFile(tempFile, 13)
    np.savetxt(dstDIR + '/' + f[0:8] + '_MFCC.txt', resultMFCC)
'''

'''
trainingDIR = "F:/Riku_Ka/Training/Depression Data for Training"
dstDIR = "F:/Riku_Ka/Training/MFCC for Depression Data for Training"
files = os.listdir(trainingDIR)
print(files)
for f in files:
    print(f)
    tempFile = trainingDIR + '/' + f
    resultMFCC =  FHC_Audio.getMFCC4SingleFile(tempFile, 13)
    np.savetxt(dstDIR + '/' + f[0:8] + '_MFCC.txt', resultMFCC)
'''


'''
RootDIR = "F:/Riku_Ka/hearlth-data"
subDIRs = os.listdir(RootDIR)

trainingDIR = "F:/Riku_Ka/Depression Data Pre Step 2"
files = os.listdir(trainingDIR)

resultFolder = "F:/Riku_Ka/Depression Data Pre Step 3"

for subdir in subDIRs:
    tempDIR = RootDIR + '/' + subdir
    
    files = os.listdir(tempDIR)
    if len(files) != 2:
        print(tempDIR)
'''


'''
for f in files:
    tempFile = trainingDIR + f
    if tempFile.find('ica2') == 1:
        os.remove(tempFile)
'''

'''
targetFile = "F:/Riku_Ka/testdata/AB001-1_voice/AB001-1_Pt_free talk_20171026_110943_071.wav"
targetFile2 = "F:/Riku_Ka/Training Data Depression/AD003-2_ica1.wav"
voice = AudioSegment.from_file(targetFile2, "wav")
print(voice.dBFS)
print(voice.rms)
print(voice.apply_gain(-voice.max_dBFS))
'''


'''
for subdir in subDIRs:
    tempDIR = RootDIR + '/' + subdir
    
    files = os.listdir(tempDIR)
    if len(files) != 4:
        print(tempDIR)
'''

'''
for f in files:
    print(f)
    FHC_Audio.VolumeUpByMultiplication(trainingDIR + '/' + f, 2)
'''

'''
for subdir in subDIRs:
    tempDIR = RootDIR + '/' + subdir
    
    files = os.listdir(tempDIR)
    for f in files:
        if f.find('ica') != -1:
           os.rename(tempDIR+'/'+f, tempDIR+'/'+subdir[0:8]+f[9:])
'''

           

'''
    
for f in files:
    tempFile = trainingDIR + '/' + f
    print(f)
    FHC_Audio.stripSilenceFromWaveFile(tempFile, resultFolder, min_silence_len=400, silence_thresh=-70)

'''