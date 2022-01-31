from collections import defaultdict
from itertools import chain, combinations

from dbconnect import *


# load variable number of playlists from our database and return a list of playlists
# and a set of all tracks

# def getFromDB(maxPlaylists):
#     playlists = []
#     unique_tracks = set()
#
#     for i in range(0, maxPlaylists):
#         tracks = dbReq(f'{"select track_uri from pcont where pid="}{i}')
#         record = set()
#         for track in tracks:
#             record.add(track)
#         for item in record:
#             unique_tracks.add(frozenset([item]))
#         playlists.append(record)
#     return unique_tracks, playlists

def getFromDB(maxPlaylists):
    maxPlaylists -= 1
    playlists = []
    unique_tracks = set()
    playlistCounter = 0
    record = set()

    tracks = dbReq(f'{"select track_uri, pid from pcont where pid<="}{maxPlaylists}')
    for track in tracks:
        if playlistCounter != track[1]:
            playlistCounter += 1
            playlists.append(record)
            record.clear()
        track_uri = tuple([track[0]])
        record.add(track_uri)
        unique_tracks.add(frozenset([track_uri]))
    playlists.append(record)
    return unique_tracks, playlists



# likelihood of a track being included in a playlist
# e.g. for minSup = 0.5 this function returns all tracks which are
#       included in a playlist with a likelihood of 50%
def getAboveMinSup(unique_tracks, playlists, minSup, globalTrackSetWithSup):
    freqTrackSet = set()
    localTrackSetWithSup = defaultdict(int)

    for track in unique_tracks:
        for playlistSet in playlists:
            if track.issubset(playlistSet):
                globalTrackSetWithSup[track] += 1
                localTrackSetWithSup[track] += 1
    for track, supCount in localTrackSetWithSup.items():
        support = float(supCount / len(playlists))
        if(support >= minSup):
            freqTrackSet.add(track)

    return freqTrackSet



def getUnion(trackSet, length):
    return set([i.union(j) for i in trackSet for j in trackSet if len(i.union(j)) == length])



def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if(frozenset(subset) not in prevFreqSet):
                tempCandidateSet.remove(item)
                break
    return tempCandidateSet



def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))



def associationRule(freqItemSet, itemSetWithSup, minConf):
    rules = []
    for k, itemSet in freqItemSet.items():
        for item in itemSet:
            subsets = powerset(item)
            for s in subsets:
                confidence = float(
                    itemSetWithSup[item] / itemSetWithSup[frozenset(s)])
                if(confidence > minConf):
                    rules.append([set(s), set(item.difference(s)), confidence])
    return rules



def aprioriFromDB(numberPlaylists, minSup = 0.5, minConf = 0.5):
    unique_tracks, playlists = getFromDB(numberPlaylists)
    # i = 0
    # for playlist in playlists:
    #     print(i, " ", playlist)
    #     i += 1
    # i = 0
    # for unique_track in unique_tracks:
    #     print(i, " ", unique_track)
    #     i += 1
    # print(unique_tracks)
    # print(playlists)

    # Final result global frequent itemset
    globalFreqTrackSet = dict()
    # Storing global itemset with support count
    globalTrackSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(
        unique_tracks, playlists, minSup, globalTrackSetWithSup)
    currentLSet = L1ItemSet
    k = 2
    #print(currentLSet)

    # Calculating frequent track set
    while (currentLSet):
        # Storing frequent itemset
        globalFreqTrackSet[k - 1] = currentLSet
        # Self-joining Lk
        candidateSet = getUnion(currentLSet, k)
        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k - 1)
        # Scanning itemSet for counting support
        currentLSet = getAboveMinSup(
            candidateSet, playlists, minSup, globalTrackSetWithSup)
        k += 1

    rules = associationRule(globalFreqTrackSet, globalTrackSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])

    return globalFreqTrackSet, rules



globalFreqTrackSet, rules = aprioriFromDB(2, 0.1, 0.5)
print(globalFreqTrackSet)
print(rules)