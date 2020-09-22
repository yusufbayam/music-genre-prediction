import numpy as np
import os
import sklearn
from sklearn.preprocessing import StandardScaler
import librosa
import librosa.display
from scipy import stats


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


genres = 'D:\genres\genres\genres'
os.chdir(genres)
for filename in os.listdir(os.getcwd()):
    if not filename.endswith(".mf"):
        os.chdir(filename)
        for songs in os.listdir(os.getcwd()):
            with open(os.path.join(os.getcwd(), songs), 'r') as f:
                print(f.name)
        os.chdir(genres)





def calculate_features(song):
    y, sr = librosa.load(song)

    features = []
    ###1 TEMPO
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    features.append(tempo)
    features
    ###2 SPECTRAL CENTROID
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spectral_centroid, axis=1))
    features.append(np.std(spectral_centroid, axis=1))
    features.append(stats.skew(spectral_centroid, axis=1))
    features.append(np.min(spectral_centroid, axis=1))
    features.append(np.max(spectral_centroid, axis=1))

    ###3 SPECTRAL ROLLOF
    spectral_rollof = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(spectral_rollof, axis=1))
    features.append(np.std(spectral_rollof, axis=1))
    features.append(stats.skew(spectral_rollof, axis=1))
    features.append(np.min(spectral_rollof, axis=1))
    features.append(np.max(spectral_rollof, axis=1))
    ###4 SEPCTRAL NOVELTY
    spectral_novelty = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(np.mean(spectral_novelty, axis=1))
    features.append(np.std(spectral_novelty, axis=1))
    features.append(stats.skew(spectral_novelty, axis=1))
    features.append(np.min(spectral_novelty, axis=1))
    features.append(np.max(spectral_novelty, axis=1))

    ###5 SPECTRAL BANDWITH
    spectral_bandwidth_2 = np.array(librosa.feature.spectral_bandwidth(y+0.01, sr=sr)[0])
    features.append(np.mean(spectral_bandwidth_2, axis=1))
    features.append(np.std(spectral_bandwidth_2, axis=1))
    features.append(stats.skew(spectral_bandwidth_2, axis=1))
    features.append(np.min(spectral_bandwidth_2, axis=1))
    features.append(np.max(spectral_bandwidth_2, axis=1))
    spectral_bandwidth_3 = np.array(librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=3)[0])
    features.append(np.mean(spectral_bandwidth_3, axis=1))
    features.append(np.std(spectral_bandwidth_3, axis=1))
    features.append(stats.skew(spectral_bandwidth_3, axis=1))
    features.append(np.min(spectral_bandwidth_3, axis=1))
    features.append(np.max(spectral_bandwidth_3, axis=1))

    ###6 SPECTRAL CONTRAST
    spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr)
    features.append(np.mean(spectral_contrast, axis=1))
    features.append(np.std(spectral_contrast, axis=1))
    features.append(stats.skew(spectral_contrast, axis=1))
    features.append(np.min(spectral_contrast, axis=1))
    features.append(np.max(spectral_contrast, axis=1))

    ###7
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    features.append(np.mean(zero_crossing_rate, axis=1))
    features.append(np.std(zero_crossing_rate, axis=1))
    features.append(stats.skew(zero_crossing_rate, axis=1))
    features.append(np.min(zero_crossing_rate, axis=1))
    features.append(np.max(zero_crossing_rate, axis=1))

    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))

    chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    features.append(np.mean(chroma_cqt, axis=1))
    features.append(np.std(chroma_cqt, axis=1))
    features.append(stats.skew(chroma_cqt, axis=1))
    features.append(np.min(chroma_cqt, axis=1))
    features.append(np.max(chroma_cqt, axis=1))

    chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    features.append(np.mean(chroma_cens, axis=1))
    features.append(np.std(chroma_cens, axis=1))
    features.append(stats.skew(chroma_cens, axis=1))
    features.append(np.min(chroma_cens, axis=1))
    features.append(np.max(chroma_cens, axis=1))

    tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)
    features.append(np.mean(tonnetz, axis=1))
    features.append(np.std(tonnetz, axis=1))
    features.append(stats.skew(tonnetz, axis=1))
    features.append(np.min(tonnetz, axis=1))
    features.append(np.max(tonnetz, axis=1))

    chroma_stft = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    features.append(np.mean(chroma_stft, axis=1))
    features.append(np.std(chroma_stft, axis=1))
    features.append(stats.skew(chroma_stft, axis=1))
    features.append(np.min(chroma_stft, axis=1))
    features.append(np.max(chroma_stft, axis=1))

    rmse = librosa.feature.rmse(S=stft)
    features.append(np.mean(rmse, axis=1))
    features.append(np.std(rmse, axis=1))
    features.append(stats.skew(rmse, axis=1))
    features.append(np.min(rmse, axis=1))
    features.append(np.max(rmse, axis=1))

    melspectrogram = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    features.append(np.mean(melspectrogram, axis=1))
    features.append(np.std(melspectrogram, axis=1))
    features.append(stats.skew(melspectrogram, axis=1))
    features.append(np.min(melspectrogram, axis=1))
    features.append(np.max(melspectrogram, axis=1))

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram), n_mfcc=20)
    features.append(np.mean(mfcc, axis=1))
    features.append(np.std(mfcc, axis=1))
    features.append(stats.skew(mfcc, axis=1))
    features.append(np.min(mfcc, axis=1))
    features.append(np.max(mfcc, axis=1))

