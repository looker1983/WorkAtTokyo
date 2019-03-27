from __future__ import division

"""
Independent component analysis (ICA) is used to estimate 
sources given noisy measurements. Imagine 2 persons speaking 
simultaneously and 2 microphones recording the mixed signals. 
ICA is used to recover the sources ie. what is said by each person.
"""

import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# loading wav files
fs_1, voice_1 = wavfile.read(
                    "C:/testData/seprate_wave/outpu_1.wav")
fs_2, voice_2 = wavfile.read(
                    "C:/testData/seprate_wave/outpu_4.wav")
# reshaping the files to have same size
m, = voice_1.shape 
voice_2 = voice_2[:m]

# plotting time domain representation of signal
figure_1 = plt.figure("Original Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, voice_1)
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of voice_2")
plt.plot(np.arange(m)/fs_2, voice_2)
plt.xlabel("Time")
plt.ylabel("Signal")

# mix data
voice = np.c_[voice_1, voice_2]
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(voice, A)

# plotting time domain representation of mixed signal
figure_2 = plt.figure("Mixed Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of mixed voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, X[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of mixed voice_2")
plt.plot(np.arange(m)/fs_2, X[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")

# blind source separation using ICA
ica = FastICA()
print("Training the ICA decomposer .....")
t_start = time.time()
ica.fit(X)
t_stop = time.time() - t_start
print("Training Complete; took %f seconds" % (t_stop))
# get the estimated sources
S_ = ica.transform(X)
# get the estimated mixing matrix
A_ = ica.mixing_
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# plotting time domain representation of estimated signal
figure_3 = plt.figure("Estimated Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of estimated voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, S_[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of estimated voice_2")
plt.plot(np.arange(m)/fs_2, S_[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")

plt.show()