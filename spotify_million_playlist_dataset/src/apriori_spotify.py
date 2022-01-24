from collections import defaultdict

from dbconnect import *


# load variable number of playlists from our database and return a list of playlists
# and a set of all tracks
def getFromDB(maxPlaylists):
    playlists = []
    unique_tracks = set()

    for i in range(0, maxPlaylists):
        tracks = dbReq(f'{"select track_uri from pcont where pid="}{i}')
        record = set()
        for track in tracks:
            record.add(track)
        for item in record:
            unique_tracks.add(frozenset([item]))
        playlists.append(record)
    return unique_tracks, playlists


# likelihood of a track being included in a playlist
# e.g. for minSup = 0.5 this function returns all tracks which are
#       included in a playlist with a likelihood of 50%
def getAboveMinSup(unique_tracks, playlists, minSup, globalItemSetWithSup):
    freqTrackSet = set()
    localTrackSetWithSup = defaultdict(int)

    for track in unique_tracks:
        for playlistSet in playlists:
            if track.issubset(playlistSet):
                globalItemSetWithSup[track] += 1
                localTrackSetWithSup[track] += 1

    for track, supCount in localTrackSetWithSup.items():
        support = float(supCount / len(playlists))
        if(support >= minSup):
            freqTrackSet.add(track)

    return freqTrackSet



def aprioriFromFile(minSup = 0.5, minConf = 0.5):
    unique_tracks, playlists = getFromDB(1)
    print(unique_tracks)
    print(playlists)

    # Final result global frequent itemset
    globalFreqItemSet = dict()
    # Storing global itemset with support count
    globalItemSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(
        unique_tracks, playlists, minSup, globalItemSetWithSup)
    currentLSet = L1ItemSet
    k = 2
    print(currentLSet)


aprioriFromFile(0.2, 0.5)