import csv
import pandas as pd


def makeTracksUnique():
    tracks = pd.read_csv("tracks_slice.csv")
    # print(tracks)
    tracks = tracks.drop_duplicates()
    print("Unique tracks ", tracks.shape[0])

def makeAlbumsUnique():
    albums = pd.read_csv("album_slice.csv")
    # print(albums)
    albums = albums.drop_duplicates('album_uri')
    print("Unique albums ", albums.shape[0])

def makeArtistUnique():
    artists = pd.read_csv("artist_slice.csv")
    # print(artists)
    artists = artists.drop_duplicates('artist_uri')
    print("Unique artists ", artists.shape[0])

makeTracksUnique()
makeAlbumsUnique()
makeArtistUnique()