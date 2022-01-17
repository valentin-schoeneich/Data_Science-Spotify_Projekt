import sys
import json
import os
import pandas as pd
import re

path = '../data/'
quick = True
max_files = 1

setHeader = True


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def getCounter():
    return f'{"_"}{str(max_files + 1)}' if quick else "_1000"


def makeCSVUnique(filename):
    print('dropping duplicates in ' + filename)
    df = pd.read_csv(f'{filename}{getCounter()}{".csv"}')
    df = df.drop_duplicates()
    df.to_csv(f'{filename}{getCounter()}{".csv"}', index=False)


def createIndex(filename):
    df = pd.read_csv(f'{filename}{getCounter()}{".csv"}')
    df.to_csv(f'{filename}{getCounter()}{".csv"}')


def process_mpd():
    global setHeader
    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames, key=natural_keys):
        if filename.startswith('mpd.slice.') and filename.endswith(".json"):
            # read json-File
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

            df_pConT = pd.json_normalize(
                data['playlists'],
                record_path=['tracks'],
                meta=[
                    'pid'
                ]
            )

            df_playlists = df_playlists.drop(['tracks'], axis=1)

            track_uri = df_pConT['track_uri']
            album_uri = df_pConT['album_uri']
            album_name = df_pConT['album_name']
            artist_uri = df_pConT['artist_uri']
            artist_name = df_pConT['artist_name']
            track_name = df_pConT['track_name']
            duration_ms = df_pConT['duration_ms']

            df_tracks = pd.concat([track_uri, track_name, duration_ms, artist_uri, album_uri], axis=1)
            df_tracks = df_tracks.drop_duplicates('track_uri')

            df_album = pd.concat([album_uri, album_name], axis=1)
            df_album = df_album.drop_duplicates('album_uri')

            df_artist = pd.concat([artist_uri, artist_name], axis=1)
            df_artist = df_artist.drop_duplicates('artist_uri')

            df_pConT = df_pConT.drop(['artist_name', 'track_name', 'duration_ms',
                                      'album_name', 'album_uri', 'artist_uri'], axis=1)

            if count > 0:
                setHeader = False

            df_playlists.to_csv(f'{"playlists"}{getCounter()}{".csv"}'
                                , mode='a', index=False, header=setHeader)  # append with mode='a', header=False
            df_artist.to_csv(f'{"artists"}{getCounter()}{".csv"}'
                             , mode='a', index=False, header=setHeader)
            df_tracks.to_csv(f'{"tracks"}{getCounter()}{".csv"}'
                             , mode='a', index=False, header=setHeader)
            df_album.to_csv(f'{"albums"}{getCounter()}{".csv"}'
                            , mode='a', index=False, header=setHeader)
            df_pConT.to_csv(f'{"pConT"}{getCounter()}{".csv"}'
                            , mode='a', index=False, header=setHeader)

            print(count, ", ", filename)

            count += 1

            if quick and count > max_files:
                break


process_mpd()
makeCSVUnique('albums')
makeCSVUnique('tracks')
makeCSVUnique('artists')
createIndex('pConT')
