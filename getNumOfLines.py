import os
import librosa

resDIR = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/MFCC with Labels for Dementia"
dstDIR = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/Num of MFCC lines for Dementia"

if not os.path.exists(dstDIR):
    os.mkdir(dstDIR)

files = os.listdir(resDIR)

for f in files:
    resFILE = resDIR + '/' + f
    dstFILE = dstDIR + '/' + "result.txt"
    print(resFILE)
    with open(resFILE, 'r', errors='ignore') as of1, open(dstFILE, 'a', errors='ignore') as of2:
        myCount = 1    
        for myCount, line in enumerate(of1):
            pass
        myCount += 1
        myList = [f,myCount]
        of2.write(str(myList[0]) + ',' + str(myList[1]) + '\n')  
    duration = librosa.get_duration()