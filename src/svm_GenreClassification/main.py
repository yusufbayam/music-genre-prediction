#                   “Music is nothing else but wild sounds civilized into time and tune.”
#                                              Thomas Fuller

import numpy as np
import os
import sklearn
from numpy import random
from sklearn.preprocessing import StandardScaler
import librosa
import librosa.display
from scipy import stats
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Embedding
from keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils.np_utils import to_categorical

timeseries_length = 1200
hop_length = 512

genre_name_list = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def calculate_features(song):
    y, sr = librosa.load(song)
    # print("song")
    # print(np.shape(y))
    data = np.zeros(
        (timeseries_length, 64), dtype=np.float64
    )

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, hop_length=hop_length, n_mfcc=13
    )
    # print(np.shape(mfcc))
    # print(mfcc)
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
    # print("tonnetz: ",np.shape(tonnetz))

    # librosa.feature.spectral_rolloff()
    # librosa.feature.zero_crossing_rate()
    # librosa.feature.tonnetz()

    data[0:timeseries_length, 0:13] = mfcc.T[0:timeseries_length, :]
    data[:timeseries_length, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[:timeseries_length, 14:26] = chroma.T[0:timeseries_length, :]
    data[:timeseries_length, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    data[:timeseries_length, 33:45] = chroma_cqt.T[0:timeseries_length, :]
    data[:timeseries_length, 45:57] = chroma_cens.T[0:timeseries_length, :]
    data[:timeseries_length, 57:58] = rmse.T[0:timeseries_length, :]
    data[:timeseries_length, 58:64] = tonnetz.T[0:timeseries_length, :]

    return data


def minimize_data(arr):
    return arr[:200, :timeseries_length, :]


all_features = []
all_labels = []
label = 0
genres = "D:\Downloads\genres"

os.chdir(genres)

if os.path.isfile('alllabels.npy'):
    print('yes')
    all_features = np.load('allfeatures.npy')
    all_features = minimize_data(all_features)
    print(np.shape(all_features))
    all_labels = np.load('alllabels.npy')
    print(np.shape(all_labels))
    all_labels = all_labels[:200]
    print(np.shape(all_labels))

else:
    print('no')
    inc = 0
    for filename in os.listdir(os.getcwd()):
        if not filename.endswith(".mf"):
            os.chdir(filename)
            print(filename)
            for songs in os.listdir(os.getcwd()):
                with open(os.path.join(os.getcwd(), songs), 'r') as f:
                    all_features.append(calculate_features(f.name))
                    all_labels.append(label)
                # inc+=1
                # if (inc%2==0):
                #     break
            label += 1
            os.chdir(genres)

    print(np.shape(all_features))

    np.save('allfeatures.npy', all_features)
    np.save('alllabels.npy', all_labels)
    print('datasaved')

c = list(zip(all_features, all_labels))
random.shuffle(c)
all_features, all_labels = zip(*c)
all_labels = to_categorical(all_labels)
all_features = np.array(all_features)
all_labels = np.array(all_labels)
# print(all_labels)

trainData = all_features[:150]
testData = all_features[150:]
trainLabels = all_labels[:150]
testLabels = all_labels[150:]

model = Sequential()
model.add(LSTM(units=64, dropout=0.35, recurrent_dropout=0.35, return_sequences=False, input_shape=(timeseries_length, 64)))
# model.add(LSTM(units=8, dropout=0.35, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=all_labels.shape[1], activation="softmax"))
print("Compiling.")

opt = Adam(lr=0.01)
# opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 100  # num of training examples per minibatch
num_epochs = 20

print(np.shape(trainData))
print(np.shape(trainLabels))

model.fit(trainData, trainLabels, epochs=num_epochs, batch_size=batch_size, validation_data=(testData, testLabels))

print("\nTesting ...")
score, accuracy = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
