import numpy as np
import librosa.display
import sklearn
import soundfile as sf
import matplotlib.pyplot as plt
import os
import random


def createSameGenreSampleList(genre):
    return ['D:\\Downloads\\genres\\{}\\{}.00001.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00007.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00022.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00024.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00029.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00035.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00041.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00039.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00061.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00033.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00073.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00065.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00011.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00056.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00043.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00067.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00071.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00084.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00092.wav'.format(genre, genre),
            'D:\\Downloads\\genres\\{}\\{}.00097.wav'.format(genre, genre)]


def createSampleSongList():
    random.seed(6)
    return ['D:\\Downloads\\genres\\blues\\blues.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\blues\\blues.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\classical\\classical.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\classical\\classical.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\country\\country.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\country\\country.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\disco\\disco.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\disco\\disco.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\hiphop\\hiphop.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\hiphop\\hiphop.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\jazz\\jazz.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\jazz\\jazz.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\metal\\metal.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\metal\\metal.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\pop\\pop.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\pop\\pop.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\reggae\\reggae.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\reggae\\reggae.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\rock\\rock.000{}.wav'.format(random.randint(10, 100)),
            'D:\\Downloads\\genres\\rock\\rock.000{}.wav'.format(random.randint(10, 100))]


def plotList(songList):
    plt.figure(figsize=(60, 25))
    for i in range(len(songList)):
        plt.subplot(5, 4, i + 1)
        y, sr = librosa.load(songList[i])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        librosa.display.specshow(mfcc, x_axis='time')
        plt.title(songList[i][20:24])
    plt.show()


timeseries_length = 100
num_of_features = 91
hop_length = 512
song_length = 661500

genres = 'D:\Downloads\genres'
os.chdir(genres)
for filename in os.listdir(os.getcwd()):
    if not (filename.endswith(".mf")) or (filename.endswith(".npy")):
        os.chdir(filename)
        print(filename)
        song_count = 0
        for songs in os.listdir(os.getcwd()):
            with open(os.path.join(os.getcwd(), songs), 'r') as f:
                print(f.name, " song_count : ", song_count)
                y, sr = librosa.load(f.name)
                song1 = y[0:132300]
                song2 = y[132300: 264600]
                song3 = y[264600:396900]
                song4 = y[396900:529200]
                song5 = y[529200:661500]
                sf.write(f'D:\\Downloads\\genres\\{filename}\\{filename+"_new"}{song_count}.wav', song1, sr)
                sf.write(f'D:\\Downloads\\genres\\{filename}\\{filename+"_new"}{song_count+1}.wav', song2, sr)
                sf.write(f'D:\\Downloads\\genres\\{filename}\\{filename+"_new"}{song_count+2}.wav', song3, sr)
                sf.write(f'D:\\Downloads\\genres\\{filename}\\{filename + "_new"}{song_count + 3}.wav', song4, sr)
                sf.write(f'D:\\Downloads\\genres\\{filename}\\{filename + "_new"}{song_count + 4}.wav', song5, sr)
                song_count = song_count + 5
                # split = librosa.effects.split(y, 20)
                # if np.size(split) > 1:
                #     print(np.shape(split))
        os.chdir(genres)


# sample_song_list = createSampleSongList()
# same_genre_sample_list = createSameGenreSampleList('jazz')
#
# plotList(same_genre_sample_list)

# song = 'D:\\Downloads\\genres\\country\\country.00065.wav'
# y, sr = librosa.load(song)

# mfcc = librosa.feature.mfcc(
#     y=y, sr=sr, hop_length=hop_length, n_mfcc=13
# )
# chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma')
# plt.show()
# print(np.shape(mfcc))

# plt.figure()
#
# librosa.display.waveplot(y=y, sr=sr)
# plt.xlabel("TIME(SECONDS) ==>")
# plt.ylabel("AMPLITUDE")
# plt.show()
#
# split = librosa.effects.split(y, 20)
#
# silent_parts = []
# for i in range(len(split) - 1):
#     silent_parts = np.append(silent_parts, y[(split[i])[1]:(split[i+1])[0]])
#
# print(silent_parts)
# print(np.shape(silent_parts))
# sf.write('D:\\Downloads\\deneme.wav', silent_parts, sr)
#
# n_fft = 2048
# S = librosa.stft(y, n_fft=n_fft, hop_length=512)
# print(S.shape)
# # convert to db
# D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
# max_db = np.max(abs(D))
# print(max_db)
