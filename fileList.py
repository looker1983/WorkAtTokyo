import os
import re

resDIR = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/MFCC with Labels for Health"
tarFILE = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/fileListForHealth.txt"

files = os.listdir(resDIR)


with open(tarFILE, 'w', errors='ignore') as of:
    for f in files:
        f = f.strip('1_')
        f = f.strip('_MFCC.txt')
        of.write(f+'\n')
