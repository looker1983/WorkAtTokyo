from __future__ import print_function

import wave
import matplotlib.pyplot as plt
import matplotlib.style as ms
import numpy as np
import scipy.io as io
import pyAudioAnalysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import datetime
import sys
import os

docPath = 'C:/testData/voiceTest/testing/'
fileName = 'C:/testData/voiceTest/01_MAD_1_10sec.wav'

files = os.listdir(docPath)

print("Test starts!")

for file in files:
    if not os.path.isdir(file):
        pass




print("Test ends!")