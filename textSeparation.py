import os

rootDIR = "C:/Users/fhc_workPC1/Documents/Looker/test results"
resFILE = "results_for_Dementia Prediction Model with our MFCC and scores splited at 27.txt"
dstFILE = "new_" + resFILE

with open(resFILE, 'r', errors='ignore', encoding='utf-8') as rf, open(dstFILE, 'w', errors='ignore', encoding='utf-8') as df:
    lines = rf.readlines()
    for l in lines:
        if l.find("The accuracy for file") != -1:
            l.strip("The accuracy for file")
            newLine = l.split("MFCC-012order.npy")
            df.write(newLine[0] + ',' + newLine[1])