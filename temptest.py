import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import shutil
import os
from pydub  import AudioSegment
import numpy as np

import FHC_Audio

trainingDIR = "F:/Riku_Ka/temptest"
dstDIR = "F:/Riku_Ka/Depression Data Pre Step final"
files = os.listdir(trainingDIR)
for f in files:
    tempFile = trainingDIR + '/' + f
    resultMFCC =  FHC_Audio.getMFCC4SingleFile(tempFile, 13)
    np.savetxt(dstDIR + '/' + f[0:6] + '_MFCC.txt', resultMFCC)