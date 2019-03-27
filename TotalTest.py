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


def whiten(X):
    #zero mean,for mean(): axis=-1 returns m*1
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, newaxis]
    #whiten
    A = dot(X, X.transpose())
    #Compute the eigenvalues and right eigenvectors of a square array.
    D , E = linalg.eig(A)
    #Compute the (multiplicative) inverse of a matrix.
    D2 = linalg.inv(array([[D[0], 0.0], [0.0, D[1]]], float32))
    D2[0,0] = sqrt(D2[0,0]); D2[1,1] = sqrt(D2[1,1])
    V = dot(D2, E.transpose())
    return dot(V, X), V

def _logcosh(x, alpha = 1):
    gx = tanh(alpha * x); g_x = gx ** 2; g_x -= 1.; g_x *= -alpha
    return gx, g_x.mean(axis=-1)

def do_decorrelation(W):
    #black magic
    s, u = linalg.eigh(dot(W, W.T))
    return dot(dot(u * (1. / sqrt(s)), u.T), W)

def do_fastica(X):
    n, m = X.shape; p = float(m); g = _logcosh
    #black magic
    X *= sqrt(X.shape[1])
    #create w
    W = ones((n,n), float32)
    for i in range(n): 
        for j in range(i):
            W[i,j] = random.random()
            
    #compute W
    maxIter = 200
    for ii in range(maxIter):
        gwtx, g_wtx = g(dot(W, X))
        W1 = do_decorrelation(dot(gwtx, X.T) / p - g_wtx[:, newaxis] * W)
        lim = max( abs(abs(diag(dot(W1, W.T))) - 1) )
        W = W1
        if lim < 0.0001:
            break
    return W

def show_data(T, S):
    plt.plot(T, [S[0,i] for i in range(S.shape[1])], marker="*")
    plt.plot(T, [S[1,i] for i in range(S.shape[1])], marker="o")
    plt.show()

rootDIR = "F:/Riku_Ka/totalTest-Depression/ready-data/"
subDIRs = os.listdir(rootDIR)
print("subDIRs are " + str(subDIRs))

for subDir in subDIRs: 
    print("subDir is " + subDir)
    subDirPath = os.path.join(rootDIR,subDir)
    print("The task for DIR {} begins.".format(subDirPath))
    startTime = time.time()
    files = os.listdir(subDirPath)
    


    PtWaveFile = subDirPath + '/' + files[0]
    print(PtWaveFile)
    ThWaveFile = subDirPath + '/' + files[1]
    print(ThWaveFile)

    
    sf_1, voice_Pt = wavfile.read(PtWaveFile)
    sf_2, voice_Th = wavfile.read(ThWaveFile)

    m, = voice_Pt.shape 
    voice_Th = voice_Th[:m]
    X = array([voice_Pt, voice_Th], float32)


    Dwhiten, K = whiten(X)
    W = do_fastica(Dwhiten)
    Y = dot(dot(W, K), X)

    
    outData_1,outData_2 = Y[0,:],Y[1,:] 
    nchannels = 1
    sampwidth = 2
    fs = 16000
    data_size = len(outData_1)
    framerate = int(fs)
    nframes = data_size
    comptype = "NONE"
    compname = "not compressed"
    
    outfile_1 = subDirPath + '/' + subDir[0:5] + '-my_ica1.wav'
    outwave_1 = wave.open(outfile_1, 'wb')  # 定义存储路径以及文件名
    outwave_1.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))
    for v_1 in outData_1:
        outwave_1.writeframes(struct.pack('h', int(v_1 * 64000 / 2))) 
    outwave_1.close()
   

    outfile_2 = subDirPath +  '/' + subDir[0:5] + '-my_ica2.wav'
    outwave_2 = wave.open(outfile_2, 'wb')  
    outwave_2.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))     
    for v_2 in outData_2:
        outwave_2.writeframes(struct.pack('h', int(v_2 * 64000 / 2)))
    outwave_2.close()
    

    print("The task for DIR {} ends.".format(subDirPath))
    endTime = time.time() - startTime
    print("The task take {} seconds.".format(endTime))

    




'''
    #print(files[0])
    #print(files[1])
    if len(files) == 2:
        print(subDirPath) 
        shutil.move(subDirPath, tempDIR)
        #print(len(files))
    #for file in files:
    #    print(file)
    if len(files) != 2:
        print(subDirPath)
    for file in files:
        if file[-3:] != 'wav':
            print(subDirPath)
'''