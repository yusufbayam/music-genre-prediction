import numpy as np
import os
import sklearn
import audioread
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
from keras.optimizers import Adam, SGD, Adadelta
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical
from operator import add
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

timeseries_length = 1200
num_of_features = 64
hop_length = 512


def calculate_PCA(data):
    temp = []
    print(np.shape(data))
    data = arrange_shape(data)
    print(np.shape(data))

    pca = PCA(n_components=64)
    for i in data:
        pri_comp = pca.fit_transform(i)
        temp.append(pri_comp)
    return arrange_shape(temp)


def arrange_shape(data):
    temp1 = []
    temp2 = []
    data = np.array(data)
    for i in range(data.shape[1]):
        for k in range(data.shape[0]):
            temp1.append(data[k, i, :])

        temp2.append(temp1)
        temp1 = []
    return np.array(temp2)


def concentrate_time(features):
    count = 1
    data = []
    for feature in features:
        if count % 4 == 0:
            data.append(feature)
        count += 1
    return data


def calculate_features(song):
    y, sr = librosa.load(song)
    data = np.zeros(
        (timeseries_length, num_of_features), dtype=np.float64
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
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    rmse = librosa.feature.rms(S=stft)

    tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)

    data[:timeseries_length, 0:13] = mfcc.T[0:timeseries_length, :]
    data[:timeseries_length, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[:timeseries_length, 14:26] = chroma.T[0:timeseries_length, :]
    data[:timeseries_length, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    data[:timeseries_length, 33:45] = chroma_cqt.T[0:timeseries_length, :]
    data[:timeseries_length, 45:57] = chroma_cens.T[0:timeseries_length, :]
    data[:timeseries_length, 57:58] = rmse.T[0:timeseries_length, :]
    data[:timeseries_length, 58:64] = tonnetz.T[0:timeseries_length, :]

    return concentrate_time(data)


all_features = []
all_labels = []
label = 0
genres = 'D:\Downloads\music_data'
os.chdir(genres)
if os.path.isfile('alllabels.npy'):
    print('yes')
    all_features = np.load('allfeatures.npy')
    all_labels = np.load('alllabels.npy')

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
                    print(np.shape(all_features))
                    all_labels.append(label)
                # inc+=1
                # if (inc%2==0):
                #     break
            label += 1
            os.chdir(genres)

    np.save('allfeatures.npy', all_features)
    np.save('alllabels.npy', all_labels)
    print('datasaved')

temp = []
for i in all_features:
    min_max_scaler = preprocessing.MinMaxScaler()
    temp.append(min_max_scaler.fit_transform(i))
all_features = np.array(temp)
# all_features = calculate_PCA(all_features)

c = list(zip(all_features, all_labels))
random.shuffle(c)
all_features, all_labels = zip(*c)
all_labels = to_categorical(all_labels)
all_features = np.array(all_features)
all_labels = np.array(all_labels)
print("PCA start")
print(np.shape(all_features))

trainData = all_features[:6400]
testData = all_features[6400:]
trainLabels = all_labels[:6400]
testLabels = all_labels[6400:]
timeseries_length = int(timeseries_length / 4)
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.05, return_sequences=False,
               input_shape=(timeseries_length, all_features.shape[2])))
model.add(Dense(units=all_labels.shape[1], activation="softmax"))

opt = SGD()


model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 128  # num of training examples per minibatch
num_epochs = 250

history = model.fit(trainData, trainLabels, validation_split=0.3, epochs=num_epochs, batch_size=batch_size,
                    shuffle=True)

print("\nTesting ...")
score, accuracy = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
