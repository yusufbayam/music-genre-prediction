import wave
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import librosa
from scipy.io.wavfile import read

blues_dir = 'D:\genres\genres\genres\\blues'
# for filename in os.listdir(blues_dir):
#    with open(os.path.join(blues_dir, filename), 'r') as f:
#        #print(f.name)

blues_path = 'D:\genres\genres\genres\\blues\\blues.00001.wav'
classical_path = 'D:\genres\genres\genres\classical\classical.00000.wav'
# samplerate, data = wavfile.read(blues_path)
# length = data.shape[0] / samplerate
# time = np.linspace(0., length, data.shape[0])
# plt.plot(time, data, label="Left channel")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()

y, sr = librosa.load(blues_path)
y2, sr2 = librosa.load(classical_path)

###1
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(beat_times)
###2
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

###3
spectral_rollof = librosa.feature.spectral_rolloff(y=y, sr=sr)

###4
spectral_novelty = librosa.onset.onset_strength(y=y, sr=sr)
spectral_novelty = np.array(spectral_novelty)
spectral_novelty = np.transpose(spectral_novelty)
print(spectral_novelty)

for i in spectral_novelty:
    plt.plot(i)
    plt.show()

# for i in spectral_centroid:
#     plt.plot(i)
#     plt.show()
# sp2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)
# for i in sp2:
#     plt.plot(i)
#     plt.show()

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X, y)
# print(clf.predict([[-0.8, -1]]))
# print(blues_dir)
