import pandas as pd
import os

base_genres = ['Rock', 'Pop', 'International', 'Instrumental', 'Hip-Hop', 'Folk', 'Experimental', 'Electronic']

new_dir = 'D:\\Downloads\\music_data'
data_dir = 'D:\\Downloads\\fma_small'


def move_genres(music_id, current_path):
    def_id = music_id
    music_id = music_id.lstrip("0")
    dfi = music_df.loc[music_df['music_id'] == music_id]
    print(music_id)
    if not dfi.empty:
        genre = dfi.iat[0, 1]
        genre = str(genre)
        new_genre_dir = new_dir + "\\" + genre
        if genre not in base_genres:
            base_genres.append(genre)
            os.makedirs(new_genre_dir)

        new_music_path = new_genre_dir + "\\" + def_id + ".mp3"
        os.replace(current_path, new_music_path)


df = pd.read_csv("D:\\Downloads\\fma_metadata\\tracks.csv")
genres = df['track.7']
music_ids = df['Unnamed: 0']
music_df = pd.concat([music_ids, genres], axis=1)
music_df.columns = ['music_id', 'genres']
music_df = music_df.drop([0, 1])
music_df['music_id'] = music_df['music_id'].astype('string')
music_df = music_df.dropna()

os.chdir(data_dir)
for filename in os.listdir(os.getcwd()):
    os.chdir(filename)
    for songs in os.listdir(os.getcwd()):
        f_name = ""
        with open(os.path.join(os.getcwd(), songs), 'r') as f:
            f_name = f.name
        print(f_name)
        current_id = f_name[-10:-4]
        move_genres(current_id, f_name)
    os.chdir(data_dir)

print(base_genres)
