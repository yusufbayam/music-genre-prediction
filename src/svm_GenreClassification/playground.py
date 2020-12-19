import numpy as np
import librosa.display


timeseries_length = 100
num_of_features = 91
hop_length = 512

song = 'D:\Downloads\genres\\rock\\rock.00004.wav'

y, sr = librosa.load(song)

data = np.zeros(
    (timeseries_length, num_of_features), dtype=np.float64)

mfcc = librosa.feature.mfcc(
    y=y, sr=sr, hop_length=hop_length, n_mfcc=40
)
spectral_center = librosa.feature.spectral_centroid(
    y=y, sr=sr, hop_length=hop_length
)
chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
spectral_contrast = librosa.feature.spectral_contrast(
    y=y, sr=sr, hop_length=hop_length
)
cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
rmse = librosa.feature.rms(S=stft)

tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)

zero_crossing = librosa.feature.zero_crossing_rate(y=y)
print('shape of zero crossing')
print(np.shape(zero_crossing))

spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
print('shape of spectral rolloff')
print(np.shape(spectral_rolloff))

spectral_bandwith = librosa.feature.spectral_bandwidth(y=y, sr=sr)
print('shape of spectral bandwith')
print(np.shape(spectral_bandwith))

spectral_flatness = librosa.feature.spectral_flatness(y=y)
print('shape of spectral_flatness')
print(np.shape(spectral_flatness))

melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
print('shape of melspectrogram')
print(np.shape(melspectrogram))

data[:timeseries_length, 0:40] = mfcc.T[0:timeseries_length, :]
data[:timeseries_length, 40:41] = spectral_center.T[0:timeseries_length, :]
data[:timeseries_length, 41:53] = chroma.T[0:timeseries_length, :]
data[:timeseries_length, 53:60] = spectral_contrast.T[0:timeseries_length, :]
data[:timeseries_length, 60:72] = chroma_cqt.T[0:timeseries_length, :]
data[:timeseries_length, 72:84] = chroma_cens.T[0:timeseries_length, :]
data[:timeseries_length, 84:85] = rmse.T[0:timeseries_length, :]
data[:timeseries_length, 85:91] = tonnetz.T[0:timeseries_length, :]
data[:timeseries_length, 91:92] = zero_crossing.T[0:timeseries_length, :]
data[:timeseries_length, 92:93] = spectral_rolloff.T[0:timeseries_length, :]
data[:timeseries_length, 93:94] = spectral_bandwith.T[0:timeseries_length, :]
data[:timeseries_length, 94:95] = spectral_flatness.T[0:timeseries_length, :]
data[:timeseries_length, 95:223] = melspectrogram.T[0:timeseries_length, :]
