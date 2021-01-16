import numpy as np
import os
import librosa
import librosa.display
from keras.optimizers import Adam
from tensorflow.python.keras.models import model_from_json
from sklearn import preprocessing

timeseries_length = 128
num_of_features = 40
hop_length = 512


def calculate_features(song):
    data = np.zeros(
        (timeseries_length, num_of_features), dtype=np.float64
    )

    mfcc = librosa.feature.mfcc(
        y=song, sr=sr, n_mfcc=40
    )

    data[:timeseries_length, 0:40] = mfcc.T[0:timeseries_length, :]

    return data


predict_samples = 'D:\Downloads\predict_samples'
os.chdir(predict_samples)

testData = []

for songs in os.listdir(os.getcwd()):
    if songs.endswith(".wav"):
        with open(os.path.join(os.getcwd(), songs), 'r') as f:
            print(f.name)
            y, sr = librosa.load(f.name)
            y, index = librosa.effects.trim(y)
            y = y[:sr*15]
            while y.size >= sr*3:
                three_sec_split = y[:sr*3]
                print(np.shape(three_sec_split))
                y = y[sr*3:]
                testData.append(calculate_features(three_sec_split))

testData = np.array(testData)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# all_features = scaler.fit_transform(all_features.reshape(all_features.shape[0], -1)).reshape(all_features.shape)
print(np.shape(testData))

opt = Adam(learning_rate=0.001)
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
print(predictData)
