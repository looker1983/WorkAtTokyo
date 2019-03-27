import math
import random
import matplotlib.pyplot as plt
from numpy import *
from scipy.io import wavfile
import wave
import struct


def whiten(X):
    #zero mean
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, newaxis]
    #whiten
    A = dot(X, X.transpose())
    D , E = linalg.eig(A)
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

def main():
    
    #n_components = 2

    samplingRate_1, data_1 = wavfile.read('C:/testData/seprate_wave/outpu_1.wav')
    samplingRate_2, data_2 = wavfile.read('C:/testData/seprate_wave/outpu_2.wav')
    X = array([data_1, data_2], float32)

    Dwhiten, K = whiten(X)
    W = do_fastica(Dwhiten)
    Sr = dot(dot(W, K), X)

    #展示混合信号及分离后的信号
    show_data(range(len(data_1)), X)
    show_data(range(len(data_1)), Sr)

    filepath = './data_sound/'
    outData_1,outData_2 = Sr[0,:],Sr[1,:]   #待写入wav的数据
    nchannels = 1
    sampwidth = 2
    fs = 8000
    data_size = len(outData_1)
    framerate = int(fs)
    nframes = data_size
    comptype = "NONE"
    compname = "not compressed"

    outfile_1 = filepath+'my_ica1.wav'
    outwave_1 = wave.open(outfile_1, 'wb')  # 定义存储路径以及文件名
    outwave_1.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))
    
    for v_1 in outData_1:
        outwave_1.writeframes(struct.pack('h', int(v_1 * 64000 / 2))) 
    outwave_1.close()
   
    outfile_2 = filepath+'my_ica2.wav'
    outwave_2 = wave.open(outfile_2, 'wb')  
    outwave_2.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))
        
    for v_2 in outData_2:
        outwave_2.writeframes(struct.pack('h', int(v_2 * 64000 / 2)))
    outwave_2.close()