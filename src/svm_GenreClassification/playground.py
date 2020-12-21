import numpy as np
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

timeseries_length = 100
num_of_features = 91
hop_length = 512

# genres = 'D:\Downloads\genres'
# os.chdir(genres)
# for filename in os.listdir(os.getcwd()):
#     if not filename.endswith(".mf"):
#         os.chdir(filename)
#         print(filename)
#         for songs in os.listdir(os.getcwd()):
#             with open(os.path.join(os.getcwd(), songs), 'r') as f:
#                 y, sr = librosa.load(f.name)
#                 split = librosa.effects.split(y, 20)
#                 if np.size(split) > 1:
#                     print(np.shape(split))
#         os.chdir(genres)

song = 'D:\\Downloads\\genres\\country\\country.00065.wav'
y, sr = librosa.load(song)

mfcc = librosa.feature.mfcc(
    y=y, sr=sr, hop_length=hop_length, n_mfcc=1
)
print(np.shape(mfcc))

split = librosa.effects.split(y, 20)

silent_parts = []
for i in range(len(split) - 1):
    silent_parts = np.append(silent_parts, y[(split[i])[1]:(split[i+1])[0]])

print(silent_parts)
print(np.shape(silent_parts))
sf.write('D:\\Downloads\\deneme.wav', silent_parts, sr)

n_fft = 2048
S = librosa.stft(y, n_fft=n_fft, hop_length=512)
print(S.shape)
# convert to db
D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
max_db = np.max(abs(D))
print(max_db)

plt.figure()

librosa.display.waveplot(y=y, sr=sr)
plt.xlabel("TIME(SECONDS) ==>")
plt.ylabel("AMPLITUDE")
plt.show()
