import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment

import FHC_Audio

RootDIR = "F:/Riku_Ka/Analyzed Data Depression"
subDIRs = os.listdir(RootDIR)

trainingDIR = "F:/Riku_Ka/Depression Data Pre Step 2"
files = os.listdir(trainingDIR)


    
for f in files:
    tempFile = trainingDIR + '/' + f
    print(f)
    FHC_Audio.stripSilenceFromWaveFile(tempFile, min_silence_len=400, silence_thresh=-72)


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

           

