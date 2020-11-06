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
from keras.layers import Dense
from keras.optimizers import Adam

timeseries_length = 128
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
    data = np.zeros(
        (timeseries_length, 33), dtype=np.float64
    )

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, hop_length=hop_length, n_mfcc=13
    )
    spectral_center = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=hop_length
    )

    data[:, 0:13] = mfcc.T[0:timeseries_length, :]
    data[:, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[:, 14:26] = chroma.T[0:timeseries_length, :]
    data[:, 26:33] = spectral_contrast.T[0:timeseries_length, :]

    #print(np.shape(data))
    return data

all_features = []
all_labels = []
label = 0
genres = 'D:\Downloads\genres'
os.chdir(genres)
for filename in os.listdir(os.getcwd()):
    if not filename.endswith(".mf"):
        os.chdir(filename)
        print(filename)
        for songs in os.listdir(os.getcwd()):
            with open(os.path.join(os.getcwd(), songs), 'r') as f:
                all_features.append(calculate_features(f.name))
                all_labels.append(label)
        label += 1
        os.chdir(genres)

print(np.shape(all_features))

c = list(zip(all_features, all_labels))
random.shuffle(c)
all_features, all_labels = zip(*c)
all_features = np.array(all_features)
all_labels = np.array(all_labels)
print(all_labels)

trainData = all_features[:150]
testData = all_features[150:]
trainLabels = all_labels[:150]
testLabels = all_labels[150:]

model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(128, 33)))
model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=1, activation="softmax"))
print("Compiling.")

opt = Adam(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 64  # num of training examples per minibatch
num_epochs = 25

model.fit(trainData, trainLabels, epochs=num_epochs, batch_size=batch_size, validation_data=(testData, testLabels))

print("\nTesting ...")
score, accuracy = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
