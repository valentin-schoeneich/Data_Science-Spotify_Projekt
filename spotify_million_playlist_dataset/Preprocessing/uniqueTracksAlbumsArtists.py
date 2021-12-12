import csv
import pandas as pd

def makeTracksUnique():
    track_list = list
    tracks = pd.read_csv("album.csv")
    print(tracks)
    tracks = tracks.drop_duplicates()
    print(tracks.shape[0])

makeTracksUnique()

#soul = pd.read_csv("soul.csv") # too large, use chunks
#print(soul.shape[0])