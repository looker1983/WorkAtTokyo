import pandas as pd
import os
import re
import shutil



srcDIR = "C:/Users/fhc_workPC1/Documents/Looker/MFCC data by Looker/MFCC with Labels for Health"
dstDIR = "C:/Users/fhc_workPC1/Documents/Looker/Dementia-Health-with-score/MFCC for Health"
#testFile = "C:/Users/fhc_workPC1/Documents/Looker/testData/AD001-1.txt"

#df = pd.read_table(testFile, delim_whitespace=True)
#colnames = ['intervalname', 'start', 'end', 'F0_max', 'F0_avg', 'F1', 'F2', 'F3', 'I_max', 'I_min', 'I_avg', 'HNR_max', 'HNR_min', 'HNR_avg', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7',\
             #'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'C0', 'CPPS']

resFILE = "C:/Users/fhc_workPC1/Documents/Looker/認知症ーMMSE点数-1 and 0.txt"
dstFILE1 = "C:/Users/fhc_workPC1/Documents/Looker/Dementia-Health-with-score/healhLIST.txt"
dstFILE2 = "C:/Users/fhc_workPC1/Documents/Looker/Dementia-Health-with-score/dementiaLIST.txt"

dementiaLIST = []
healthLIST = []

with open(dstFILE1, 'r') as f1, open(dstFILE2, 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    for l1 in lines1:
        l1 = l1.strip('\n')
        healthLIST.append(l1)
    for l2 in lines2:
        l2 = l2.strip('\n')
        dementiaLIST.append(l2)
    print(dementiaLIST)
    print("********************************")
    print(healthLIST)

files = os.listdir(srcDIR)


for f in files:
    if any(x in f for x in healthLIST):
        srcFILE = srcDIR + '/' + f
        dstFILE = dstDIR + '/' + f
        print("For the file {}: ".format(f))
        try:
            shutil.copy(srcFILE, dstFILE)     
        except ValueError:
            print("There is a ValueError.")

    
    #f2.write(dementiaLIST)
    #f3.write(healthLIST)

    


'''
# Delete the some columns
files= os.listdir(srcDIR)
for f in files:
    srcFILE = srcDIR + '/' + f
    dstFILE = dstDIR + '/' + f
    df = pd.read_table(srcFILE, delim_whitespace=True)
    df.drop(df.columns[0:15], axis=1,inplace=True)
    df.drop(df.columns[-1], axis=1,inplace=True)
    df.drop(df.columns[-1], axis=1,inplace=True)
    df.to_csv(dstFILE, sep='\t', index=False, header=None)
'''

'''
# Delete the rows contain "undefined"
files= os.listdir(srcDR)
for f in files:
    srcFILE = srcDIR + '/' + f
    dstFILE = dstDIR + '/' + f
    df = pd.read_table(srcFILE, delim_whitespace=True, index_col=0)
    for item in colnames:
        df = df[df[item] != "--undefined--"]
    df.to_csv(dstFILE, sep='\t')
'''













'''
print(colnames)

for item in colnames:
    df = df[df.item != "--undefined--"]


for myItem in colnames:
    df = df[df.F0_avg != "--undefined--"]
print(df)

#print(df)
#print(df.head(3))
#print(df.index)
#print(df[2:6])
print(df)
df = df[df.F0_avg != "--undefined--"]
print(df)
'''


'''
files= os.listdir(srcDIR)
for f in files:
    tempFile = srcDIR + '/' + f
    df = pd.read_table(tempFile, header=None, delim_whitespace=True, index_col=0)
'''




