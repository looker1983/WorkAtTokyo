import sys
sys.path.append("F:/Riku_Ka/AudioTestCode")
import pprint
import FHC_Audio
import math
import random
import matplotlib.pyplot as plt
from numpy import *
from scipy.io import wavfile
import wave
import struct
import datetime
import time
import os
import shutil

FHC_Audio.helloFHC()


RootDIR = "F:/Riku_Ka/Analyzed Data Depression"
dstDIR = "F:/Riku_Ka/Analyzed Data Depression/Training Data"
fileList = FHC_Audio.findFilesContainCertainWords(RootDIR, "ica")

FHC_Audio.moveFiles(fileList, dstDIR)


