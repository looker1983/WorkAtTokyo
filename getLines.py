import os

resDIR = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/MFCC with Labels for Dementias"
dstDIR = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/Partial MFCC features for Dementias"

files = os.listdir(resDIR)

for f in files:
    resFILE = resDIR + '/' + f
    dstFILE = dstDIR + '/' + f
    fileSIZE = os.path.getsize(resFILE)
    if fileSIZE > 20000000:
        with open(resFILE, 'r', errors='ignore') as of1, open(dstFILE, 'w', errors='ignore') as of2:
            newLines = of1.readlines()[1000:2000]
            of2.writelines(newLines)
    

