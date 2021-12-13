import sys
import json
import os
import pandas as pd
import re


path = '../../../../Data-Science-Spotify/spotify_million_playlist_dataset/data/'
quick = False
max_files_for_quick_processing = 1
df_playlists = pd.DataFrame({'name': [], 'collaborative': [], 'pid': [], 'modified_at': [], 'num_albums': [],
                             'num_tracks': [], 'num_followers': [], 'num_edits': [], 'duration_ms': [],
                             'num_artists': []})
df_soul = pd.DataFrame({'track_uri': [], 'artist_uri': [], 'album_uri': [], 'pos': [], 'pid': []})
df_tracks = pd.DataFrame({'track_uri': [], 'track_name': [], 'duration_ms': []})
df_album = pd.DataFrame({'album_uri': [], 'album_name': []})
df_artist = pd.DataFrame({'artist_uri': [], 'artist_name': []})
setHeader = True


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def process_mpd():
    global df_playlists, df_soul, df_tracks, df_album, df_artist, setHeader
    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames, key=natural_keys):
        if filename.startswith('mpd.slice.') and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            data = json.loads(js)

            df_playlists = pd.json_normalize(
                data,
                record_path=['playlists'],
                max_level=1
            )

            df_soul = pd.json_normalize(
                data['playlists'],
                record_path=['tracks'],
                meta=[
                    'pid'
                ]
            )

            df_playlists = df_playlists.drop(['tracks', 'description'], axis=1)

            track_uri = df_soul['track_uri']
            track_name = df_soul['track_name']
            duration_ms = df_soul['duration_ms']
            df_tracks = pd.concat([track_uri, track_name, duration_ms], axis=1)
            df_tracks = df_tracks.drop_duplicates('track_uri')

            album_uri = df_soul['album_uri']
            album_name = df_soul['album_name']
            df_album = pd.concat([album_uri, album_name], axis=1)
            df_album = df_album.drop_duplicates('album_uri')

            artist_uri = df_soul['artist_uri']
            artist_name = df_soul['artist_name']
            df_artist = pd.concat([artist_uri, artist_name], axis=1)
            df_artist = df_artist.drop_duplicates('artist_uri')

            df_soul = df_soul.drop(['artist_name', 'track_name', 'duration_ms', 'album_name'], axis=1)

            if count > 0:
                setHeader = False

            df_playlists.to_csv('playlists.csv', mode='a', index=False, header=setHeader)  # append with mode='a', header=False
            df_artist.to_csv('artists.csv', mode='a', index=False, header=setHeader)
            df_tracks.to_csv('tracks.csv', mode='a', index=False, header=setHeader)
            df_album.to_csv('albums.csv', mode='a', index=False, header=setHeader)
            df_soul.to_csv('soul.csv', mode='a', index=False, header=setHeader)

            print(count, ", ", filename)

            count += 1

            if quick and count > max_files_for_quick_processing:
                break

process_mpd()






