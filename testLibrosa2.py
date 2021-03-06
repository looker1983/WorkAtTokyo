from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display
import wave
import datetime
import sys
import os

docPath = 'C:/testData/voiceTest/testing/'
fileName = 'C:/testData/voiceTest/01_MAD_1_10sec.wav'

files = os.listdir(docPath)

print("Test starts!")

for file in files:
    if not os.path.isdir(file):
        '''
        ####################################################################
        #获取基本信息并绘制波形图
        myWave = wave.open(docPath + file, 'r')  
        # return:
        # self.getnchannels()：返回声道数，1标示单声道，2表示立体声 
        # self.getsampwidth()：采样带宽，即每一帧的字节宽度。单位为byte。比如2表示2byte，16位。
        # self.getframerate()：采样频率，比如48000表示48 kHz
        # self.getnframes()：返回帧数。
        # self.getcomptype(), 
        # self.getcompname())
        print(os.path.basename(sys.argv[0]))
        for item in enumerate(myWave.getparams()):
            print(item)
        
        num_frame = myWave.getnframes()
        num_Channel = myWave.getnchannels()
        num_framerate = myWave.getframerate()

        #readframes():从流的当前指针位置一次读出音频的n个帧，并且指针后移n个帧，返回一个字节数组。
        #readframes返回的是二进制数据,在Python中用字符串表示二进制数据.
        voiceStrData = myWave.readframes(num_frame)

        voiceNumData = np.fromstring(voiceStrData, dtype=np.short)
        voiceNumData.shape = -1,num_Channel


        time = np.arange(0, num_frame) * (1.0/num_framerate) 
        # 画声音波形
        plt.plot(time, voiceNumData, c='r')  
        plt.xlabel("time (seconds)")
        plt.savefig("example1.jpg")
        plt.show() 
        plt.pause(2)
        myWave.close()
        ####################################################################
        '''


        ####################################################################
        y,sr = librosa.load(file)

        ms.use('seaborn-muted')

        S = librosa.feature.melspectrogram(y,sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S,ref=np.max)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(log_S,sr=sr,x_axis='time',y_axis='mel')
        plt.title('mel power spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.pause(2)

        ####################################################################
        # Harmonic-percussive source separation start！
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
        S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

        log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
        log_Sp = librosa.power_to_db(S_harmonic, ref=np.max)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')
        plt.title('mel power spectrogram (Harmonic)')
        plt.colorbar(format='%+02.0f dB')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram (Percussive)')
        plt.colorbar(format='%+02.0f dB')

        plt.tight_layout()
        plt.pause(2)
        # Harmonic-percussive source separation ends！
        ####################################################################

        ####################################################################
        # Chromagram starts!
        C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        plt.figure(figsize=(12,4))

        librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

        plt.title('Chromagram')
        plt.colorbar()

        plt.tight_layout()
        plt.pause(2)
        # Chromagram ends!
        ####################################################################


        ####################################################################
        # MFCC starts!
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        delta_mfcc  = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        print(mfcc)
        print(delta_mfcc)
        print(delta2_mfcc)

        ISOTIMEFORMAT = '%Y-%m%d-%H%M'
        myTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)+".dat"
        myNewFile = open("C:/testVoice" + myTime,'w')

        myNewFile.write("The following is for mfcc.")
        myNewFile.write(str(mfcc))
        myNewFile.write("The following is for delta_mfcc.")
        myNewFile.write(str(delta_mfcc))
        myNewFile.write("The following is for delta2_mfcc.")
        myNewFile.write(str(delta2_mfcc))
        np.savetxt('C:/testNumpy.dat',mfcc.T)

        myNewFile.close()


        plt.figure(figsize=(12, 6))

        plt.subplot(3,1,1)
        librosa.display.specshow(mfcc)
        plt.ylabel('MFCC')
        plt.colorbar()

        plt.subplot(3,1,2)
        librosa.display.specshow(delta_mfcc)
        plt.ylabel('MFCC-$\Delta$')
        plt.colorbar()

        plt.subplot(3,1,3)
        librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
        plt.ylabel('MFCC-$\Delta^2$')
        plt.colorbar()

        plt.tight_layout()

        M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        #MFCC ends!
        ####################################################################




print("Test ends!")
'''
####################################################################
# Beat tracking starts!
plt.figure(figsize=(12, 6))
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

plt.vlines(librosa.frames_to_time(beats), 1, 0.5 * sr,\
colors='w', linestyles='-', linewidth=2, alpha=0.5)

plt.axis('tight')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
plt.pause(2)
# Beat tracking ends!
####################################################################
'''



