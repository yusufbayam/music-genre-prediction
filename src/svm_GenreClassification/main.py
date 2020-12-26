import numpy as np
import os
from numpy import random
import librosa
import librosa.display
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, Adadelta
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

timeseries_length = 128
num_of_features = 40
hop_length = 512

#TODO heatmap for features
#TODO Sliding window

# def calculate_PCA(data):
#     temp = []
#     print(np.shape(data))
#     data = arrange_shape(data)
#     print(np.shape(data))
#
#     pca = PCA(n_components=64)
#     for i in data:
#         pri_comp = pca.fit_transform(i)
#         temp.append(pri_comp)
#     return arrange_shape(temp)


# def arrange_shape(data):
#     temp1 = []
#     temp2 = []
#     data = np.array(data)
#     for i in range(data.shape[1]):
#         for k in range(data.shape[0]):
#             temp1.append(data[k, i, :])
#
#         temp2.append(temp1)
#         temp1 = []
#     return np.array(temp2)

def calculate_features(song):
    y, sr = librosa.load(song)
    data = np.zeros(
        (timeseries_length, num_of_features), dtype=np.float64
    )

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40
    )
    spectral_center = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=hop_length
    )
    # cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    # chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    # chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    rmse = librosa.feature.rms(S=stft)

    # tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)

    zero_crossing = librosa.feature.zero_crossing_rate(y=y)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    spectral_bandwith = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    spectral_flatness = librosa.feature.spectral_flatness(y=y)

    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    data[:timeseries_length, 0:40] = mfcc.T[500:timeseries_length+500, :]
    # data[:timeseries_length, 40:41] = spectral_center.T[0:timeseries_length, :]
    # data[:timeseries_length, 41:53] = chroma.T[0:timeseries_length, :]
    # data[:timeseries_length, 53:60] = spectral_contrast.T[0:timeseries_length, :]
    # data[:timeseries_length, 60:72] = chroma_cqt.T[0:timeseries_length, :]
    # data[:timeseries_length, 72:84] = chroma_cens.T[0:timeseries_length, :]
    # data[:timeseries_length, 60:61] = rmse.T[0:timeseries_length, :]
    # data[:timeseries_length, 61:67] = tonnetz.T[0:timeseries_length, :]
    # data[:timeseries_length, 61:62] = zero_crossing.T[0:timeseries_length, :]
    # data[:timeseries_length, 62:63] = spectral_rolloff.T[0:timeseries_length, :]
    # data[:timeseries_length, 63:64] = spectral_bandwith.T[0:timeseries_length, :]
    # data[:timeseries_length, 64:65] = spectral_flatness.T[0:timeseries_length, :]
    #data[:timeseries_length, 95:223] = melspectrogram.T[0:timeseries_length, :]

    return data

genres = 'D:\Downloads\genres'
os.chdir(genres)
if os.path.isfile('alllabels.npy'):
    print('yes')
    all_features = np.load('allfeatures.npy')
    all_labels = np.load('alllabels.npy')

else:
    print('no')
    all_features = []
    all_labels = []
    label = 0
    for filename in os.listdir(os.getcwd()):
        if not filename.endswith(".mf"):
            os.chdir(filename)
            print(filename)
            song_count = 0
            for songs in os.listdir(os.getcwd()):
                if song_count < 500:
                    with open(os.path.join(os.getcwd(), songs), 'r') as f:
                        all_features.append(calculate_features(f.name))
                        all_labels.append(int(label))
                    song_count = song_count + 1
                else:
                    break
            label = label + 1
            os.chdir(genres)

    np.save('allfeatures.npy', all_features)
    np.save('alllabels.npy', all_labels)
    print('datasaved')

c = list(zip(all_features, all_labels))
random.shuffle(c)
all_features, all_labels = zip(*c)
all_labels = to_categorical(all_labels)
all_features = np.array(all_features)
all_labels = np.array(all_labels)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# all_features = scaler.fit_transform(all_features.reshape(all_features.shape[0], -1)).reshape(all_features.shape)

trainData = all_features[:800]
testData = all_features[800:]
trainLabels = all_labels[:800]
testLabels = all_labels[800:]

model = Sequential()
model.add(LSTM(units=256, recurrent_dropout=0.01, return_sequences=True,
               input_shape=(timeseries_length, all_features.shape[2])))
model.add(LSTM(units=256, recurrent_dropout=0.01))
model.add(Dropout(0.5))
model.add(Dense(units=all_labels.shape[1], activation="softmax"))

opt = Adam(learning_rate=0.001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 256  # num of training examples per minibatch
num_epochs = 20

# define your model
history = model.fit(trainData, trainLabels, validation_split=0.125, epochs=num_epochs,
                    batch_size=batch_size, shuffle=True)

print("\nTesting ...")
score, accuracy = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
