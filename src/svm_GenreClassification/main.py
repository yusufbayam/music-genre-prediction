import numpy as np
import os
from numpy import random
import librosa
import librosa.display
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, Adadelta
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import preprocessing

timeseries_length = 128
num_of_features = 40
hop_length = 512


def calculate_features(song):
    y, sr = librosa.load(song)
    data = np.zeros(
        (timeseries_length, num_of_features), dtype=np.float64
    )

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40
    )

    data[:timeseries_length, 0:40] = mfcc.T[0:timeseries_length, :]

    return data

genres = 'D:\Downloads\genres_new'
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
print(np.shape(all_features))

trainData = all_features[:4000]
testData = all_features[4000:]
trainLabels = all_labels[:4000]
testLabels = all_labels[4000:]

opt = Adam(learning_rate=0.001)

if os.path.isfile('model.json'):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    predictData = loaded_model.predict(testData)
    print(np.shape(predictData))
    pred_labels = []
    counter = 0
    for i in range(len(predictData)):
        pred_index = np.argmax(predictData[i])
        test_index = np.argmax(testLabels[i])

        print("pred index: ", pred_index, " test index : ", test_index)

        if pred_index == test_index:
            counter = counter + 1
    acc = counter / np.size(testLabels[0])
    print("Accuracy: ", acc)
else:
    model = Sequential()
    model.add(LSTM(units=256, recurrent_dropout=0.01, return_sequences=True,
                   input_shape=(timeseries_length, all_features.shape[2])))
    model.add(LSTM(units=256, recurrent_dropout=0.01))
    model.add(Dropout(0.7))
    model.add(Dense(units=all_labels.shape[1], activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    print("Training ...")
    batch_size = 256  # num of training examples per minibatch
    num_epochs = 100

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

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")
