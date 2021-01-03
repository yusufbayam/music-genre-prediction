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
from keras.models import model_from_json
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from scipy.stats import norm, kurtosis, skew

timeseries_length = 300
num_of_features = 40
num_of_svm_features = 336
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
def feature_statistics(feature):
    statistics = []

    for i in feature:
        mean = np.mean(i)
        max = np.max(i)
        min = np.min(i)
        std = np.std(i)
        mean_gradi = np.mean(np.gradient(i))
        std_gradi = np.std(np.gradient(i))
        ske = skew(i)

        kurtosi = kurtosis(i)
        statistics.append(mean)
        statistics.append(max)
        statistics.append(min)
        statistics.append(std)
        statistics.append(kurtosi)
        statistics.append(ske)
        statistics.append(mean_gradi)
        statistics.append(std_gradi)



    return statistics

def calculate_features(song):
    y, sr = librosa.load(song)
    data = np.zeros(
        (timeseries_length, num_of_features), dtype=np.float64
    )
    svm_data = np.zeros(
        (num_of_svm_features), dtype=np.float64
    )
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, hop_length=hop_length, n_mfcc=40
    )
    spectral_center = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )
    rms = librosa.feature.rms(y=y)
    zero_crossing = librosa.feature.zero_crossing_rate(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)


    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    # spectral_contrast = librosa.feature.spectral_contrast(
    #     y=y, sr=sr, hop_length=hop_length
    # )
    # cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    # chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    # chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    # stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    # tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)
    # spectral_bandwith = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)


    mfcc_svm = feature_statistics(mfcc)
    rms_svm = feature_statistics(rms)
    spectral_center_svm = feature_statistics(spectral_center)
    zero_crossing_svm = feature_statistics(zero_crossing)
    spectral_rolloff_svm = feature_statistics(spectral_rolloff)
    spectral_flatness_svm = feature_statistics(spectral_flatness)

    print(np.shape(mfcc_svm))


    svm_data[0:320] = mfcc_svm
    svm_data[320:328] = spectral_center_svm
    svm_data[328:336] = spectral_rolloff_svm

    data[:timeseries_length, 0:40] = mfcc.T[0:timeseries_length, :]
    # data[:timeseries_length, 40:41] = spectral_center.T[0:timeseries_length, :]
    # data[:timeseries_length, 41:42] = spectral_rolloff.T[0:timeseries_length, :]


    # data[:timeseries_length, 0:40] = mfcc.T[0:timeseries_length, :]
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
    # data[:timeseries_length, 95:223] = melspectrogram.T[0:timeseries_length, :]

    return data, svm_data

genres = 'D:\genres\genres\genres'
os.chdir(genres)
if os.path.isfile('alllabels.npy'):
    print('yes')
    all_features = np.load('allfeatures.npy')
    all_labels = np.load('alllabels.npy')
    all_svm_features = np.load('allsvmfeatures.npy')

else:
    print('no')
    all_features = []
    all_svm_features = []
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
                        timeseries_data, svm_data = calculate_features(f.name)
                        all_features.append(timeseries_data)
                        all_svm_features.append(svm_data)
                        print(np.shape(all_svm_features))
                        all_labels.append(int(label))
                    song_count = song_count + 1
                else:
                    break
            label = label + 1
            os.chdir(genres)
    all_features = np.array(all_features)
    all_svm_features = np.array(all_svm_features)
    np.save('allsvmfeatures.npy', all_svm_features)
    np.save('allfeatures.npy', all_features)
    np.save('alllabels.npy', all_labels)
    print('datasaved')




c = list(zip(all_svm_features, all_features, all_labels))
random.shuffle(c)
all_svm_features, all_features, all_labels = zip(*c)

svm_labels = all_labels

all_labels = to_categorical(all_labels)
all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(np.shape(all_features))
print(np.shape(all_svm_features))

trainData = all_features[:800]
testData = all_features[800:]
trainLabels = all_labels[:800]
testLabels = all_labels[800:]

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# trainData = scaler.fit_transform(trainData.reshape(trainData.shape[0], -1)).reshape(trainData.shape)

if os.path.isfile('model.json'):
    print("model found")
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    opt = Adam()
    loaded_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])    # score, accuracy = loaded_model.evaluate(testData, testLabels, batch_size=256, verbose=1)
    lstm_probs = loaded_model.predict(testData)
    lstm_probsss = loaded_model.predict(trainData)
    # for i in lstm_probsss:
    #     print(i)


    #svm
    all_svm_features = np.array(all_svm_features)
    all_svm_features = all_svm_features[:, :320]
    svm_labels_train = svm_labels[:800]
    svm_labels_test = svm_labels[800:]
    clf = svm.SVC(decision_function_shape='ovr', kernel= 'linear' )
    all_svm_features = np.squeeze(all_svm_features)
    svm_train = all_svm_features[:800]
    svm_test = all_svm_features[800:]
    clf.fit(svm_train, svm_labels_train)

    svm_probs = clf.decision_function(svm_test)
    pred = []
    print(np.shape(svm_probs))
    print(np.shape(lstm_probs))
    count = 0
    for (i, j) in zip(svm_probs, lstm_probs):
        # print(i)
        # print(j)
        sum_list = [(a * 0) + (b ) for a, b in zip(i, j)]
        # print(sum_list)
        maximum = np.max(sum_list)
        index_of_maximum = np.where(sum_list == maximum)
        # count += 1
        # if(count == 3 ):
        #     break
        # print(index_of_maximum)
        pred.append(index_of_maximum)

    counter = 0
    cc = 0
    for i in range(len(pred)):
        if (pred[i] == svm_labels_test[i]):
            counter += 1
        else:
            # print(svm_labels_test[i], "-> ", pred[i])
            cc += 1

    print(100 * counter / 200)
    print( cc)



else:
    print("model not found")

    # create the sub-models
    #
    ####################################LSTM
    model = Sequential()
    model.add(LSTM(units=256, recurrent_dropout=0.05, return_sequences=True,input_shape=(timeseries_length, all_features.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, recurrent_dropout=0.05, return_sequences=False))
    model.add(Dense(units=all_labels.shape[1], activation="softmax"))

    opt = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    print("Training ...")
    batch_size = 256  # num of training examples per minibatch
    num_epochs = 50

    # define your model
    history = model.fit(trainData, trainLabels, validation_split=0.125, epochs=num_epochs,batch_size=batch_size, shuffle=True)
    lstm_probs = model.predict_proba(trainData)

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

    ####################################SVM
    all_svm_features = np.array(all_svm_features)
    all_svm_features = all_svm_features[:, :320]
    svm_labels_train = svm_labels[:800]
    svm_labels_test = svm_labels[800:]
    clf = svm.SVC(decision_function_shape='ovr', kernel= 'linear' )
    all_svm_features = np.squeeze(all_svm_features)
    svm_train = all_svm_features[:800]
    svm_test = all_svm_features[800:]
    clf.fit(svm_train, svm_labels_train)

    svm_probs = clf.decision_function(svm_test)
    pred = []

    for (i, j) in zip(svm_probs, lstm_probs):

        sum_list = [(a * 0) + (b ) for a, b in zip(i, j)]
        maximum = np.max(sum_list)
        index_of_maximum = np.where(sum_list == maximum)
        print(index_of_maximum)
        pred.append(index_of_maximum)

    counter = 0
    pred = np.array(pred)
    for i in range(len(pred)):
        if (pred[i] == svm_labels_test[i]):
            counter += 1
    print(100 * counter / 200)


    # #  #To save the model

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")



