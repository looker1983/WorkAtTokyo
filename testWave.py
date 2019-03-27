import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as wavfilepip 
#import python-magic-bin
import pyAudioAnalysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import datetime


print("Wave test starts!\n")
startTime = datetime.datetime.now()
#fileName = 'C:/testData/voiceTest/01_MAD_1.wav'
fileName = 'F:/Riku_Ka/testdata/AB001-1_voice/result/AB001-1_Pt_free talk1.wav'
resultFile = 'F:/Riku_Ka/testdata/AB001-1_voice/tempResult.txt'
#fileName = 'C:/testData/voiceTest/FOY0101ANT0.wav'

with wave.open(fileName, 'r') as myWave:
    # return:
    # self.getnchannels()：返回声道数，1标示单声道，2表示立体声 
    # self.getsampwidth()：采样带宽，即每一帧的字节宽度。单位为byte。比如2表示2byte，16位。
    # self.getframerate()：采样频率，比如48000表示48 kHz
    # self.getnframes()：返回帧数。
    # self.getcomptype(), 
    # self.getcompname())
    for item in enumerate(myWave.getparams()):
        print(item)
    
    num_frame = myWave.getnframes()
    num_Channel = myWave.getnchannels()
    num_framerate = myWave.getframerate()

    #readframes():从流的当前指针位置一次读出音频的n个帧，并且指针后移n个帧，返回一个字节数组。
    #readframes返回的是二进制数据,在Python中用字符串表示二进制数据.
    voiceStrData = myWave.readframes(num_frame)

voiceNumData = np.fromstring(voiceStrData, dtype=np.short)
#voiceNumData.tofile(resultFile)
print("The Num of lines of voiceNumData is: " + str(voiceNumData.shape[0]))
#print("The Num of rows of voiceNumData is: " + str(voiceNumData.shape[1]))
voiceNumData.shape = -1,num_Channel
print("The new Num of lines of voiceNumData is: " + str(voiceNumData.shape[0]))
print("The new Num of rows of voiceNumData is: " + str(voiceNumData.shape[1]))
#voiceNumData = voiceNumData.T

time = np.arange(0, num_frame) * (1.0/num_framerate) 
#time = np.delete(time, time.shape[0]-1)
#time = np.delete(time, time.shape[0]-1)
time.tofile(resultFile)
print("The Num of lines of time is: " + str(time.shape[0]))
#print("The Num of rows of time is: " + str(time.shape[1]))
# 画声音波形
plt.plot(time, voiceNumData, c='r')  
plt.xlabel("time (seconds)")
plt.savefig("audioTest.png")
plt.show() 
plt.pause(3)

'''
The following is related to pyAudioAnalysis
'''
[Fs, x] = audioBasicIO.readAudioFile(fileName)
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
plt.subplot(2,1,1)
plt.plot(F[0,:])
plt.xlabel('Frame NO.')
plt.ylabel(f_names[0]) 
plt.subplot(2,1,2)
plt.plot(F[1,:])
plt.xlabel('Frame no')
plt.ylabel(f_names[1])
plt.show()
plt.pause(3)



endTime = datetime.datetime.now()
print("\nWave test ends! The run time is " + str(endTime - startTime))
    

    
    
'''
    myParams = myWave.getparams()
    a,b,c,NumOfFrames = myParams[:4]
    print(a,b,c,NumOfFrames)

    #readframes():从流的当前指针位置一次读出音频的n个帧，并且指针后移n个帧，返回一个字节数组。
    #readframes返回的是二进制数据,在Python中用字符串表示二进制数据.
    voiceData = myWave.readframes(NumOfFrames)
    print(voiceData)

    # 帧总数
    numOfFrames = myWave.getnframes()
    print("numOfFrames = " + str(numOfFrames))
    #采样频率
    myWaveFramerate = myWave.getframerate()
    print("f = " + str(myWaveFramerate))
    # 采样点的时间间隔
    sample_time_interval = 1/myWaveFramerate               
    print("sample_time = " + str(sample_time_interval))
    #声音信号的长度
    time_span = numOfFrames/myWaveFramerate   
    print("time_span = " + str(time_span))

    audio_sequence = myWave.readframes(numOfFrames)
    print(audio_sequence)           #声音信号每一帧的“大小”
    x_seq = np.arange(0,time_span,sample_time_interval)

    plt.plot(x_seq,list(audio_sequence),'blue')
    plt.xlabel("time (s)")
    plt.show()
'''

