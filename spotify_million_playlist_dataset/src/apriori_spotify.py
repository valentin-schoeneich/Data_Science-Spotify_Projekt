import itertools
from collections import defaultdict
from itertools import chain, combinations

from db import *


# load variable number of playlists from our database
# returns unique tracks of each playlist and unique tracks of all playlist


# likelihood of a track being included in a playlist
# e.g. for minSup = 0.5 this function returns all tracks which are
#       included in a playlist with a likelihood of 50%
def getAboveMinSup(unique_tracks, playlists, minSup, globalTrackSetWithSup):
    freqTrackSet = set()
    localTrackSetWithSup = defaultdict(int)
    i = 0
    for track in unique_tracks:
        i += 1
        for playlistSet in playlists:
            if track.issubset(playlistSet):
                globalTrackSetWithSup[track] += 1
                localTrackSetWithSup[track] += 1
    for track, supCount in localTrackSetWithSup.items():
        support = float(supCount / len(playlists))
        if support >= minSup:
            freqTrackSet.add(track)
    return freqTrackSet


def getUnion(trackSet, length):
    '''
    This method was used by our first apriori-attempt. It is too slow for more than 10.000 tracks
    and returns way too much candidates in first round from l1TrackSet to l2TrackSet.
    With length = 2 it returns len(trackSet) * (len(trackSet)-1) / 2 candidates.
    :param trackSet:    The currentTrackSet with tracks above minSup
    :param length:      The length of the next trackSetItems.
    :return:    Returns a set of subset's with length of the parameter.
                E.g trackSet = {A,B,C,D}, length = 2
                returns: {{A,B}, {A,C}, {A,D}, {B,C}, {B,D}, {C,D}}
    '''
    return set([i.union(j) for i in trackSet for j in trackSet if len(i.union(j)) == length])


def getNextTrackSet(playlistsDict, tracksDict, minSupPercent, maxPlaylists, k):
    '''
    Combines getAboveMinSup() and getUnion() and is also way faster.
    :param playlistsDict:   This parameter is used to iterate over the playlists of a track to
                            concatenate it only with tracks from playlists itself appears in.
                            Due to an average occurrence of 5 playlists per track it reduces the number of candidates
                            significantly
                            This parameter is also used to get the support of a track by counting the playlists
    :param k:               Size of trackSet
    :param minSupPercent:   Specifies the minimum number of playlists the trackSet must appear in
    :param tracksDict:      This parameter is necessary to get the tracks above minSup of a playlist, so that
                            we don't have to iterate over the entire playlist
    :return:    playlistsDict and tracksDict of next trackSet in form of:
                tracksDict:
                {
                    pid: {frozenset({'track_uri', 'track_uri', ...}), ...}
                    5: {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1', ...}),
                        frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4', ...})},
                    0: {frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0', ...}),
                        frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4', ...})},
                    ...
                }
                playlistsDict:
                 {
                    frozenset({'track_uri', 'track_uri', ...}): {pid, pid, ...},
                    frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1', ...}): {0, 5},
                    frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4', ...}): {0, 5},
                    ...
                }
    '''
    minSup = maxPlaylists * minSupPercent  # the actual number of playlists the tracks must appears in
    candidates = set()
    for trackSet1 in playlistsDict.keys():
        for tS1playlist in playlistsDict.get(trackSet1):
            for trackSet2 in tracksDict.get(tS1playlist):
                tS1Playlists = playlistsDict.get(trackSet1)
                tS2Playlists = playlistsDict.get(trackSet2)
                supplyOfBoth = len(tS1Playlists.intersection(tS2Playlists))
                if supplyOfBoth >= minSup:  # both tracks are together in more or equal than minSup playlists
                    candidate = frozenset(set([trackSet1]).union(set([trackSet2])))
                    if len(candidate) == k:
                        candidates.add(candidate)
    return candidates


def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if (frozenset(subset) not in prevFreqSet):
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
                if (confidence > minConf):
                    rules.append([set(s), set(item.difference(s)), confidence])
    return rules


def printInfo(playlists, unique_tracks, trackSet):
    usableTracks = list()
    for tracks in trackSet:
        for track in tracks:
            usableTracks.append(track)
    usableTracks = set(usableTracks)
    num_playlists = len(playlists)
    num_unique_tracks = len(unique_tracks)
    num_usableTracks = len(usableTracks)

    print("tracks for recom.: ", num_usableTracks, "->", round(num_usableTracks / num_unique_tracks * 100, 2), "%")


def aprioriFromDB(maxPlaylists, minSup, minConf=0.5, kMax=2):
    uniqueTracks, playlists = getFromDB(maxPlaylists)
    numUniqueTracks = len(uniqueTracks)
    numPlaylists = len(playlists)  # number of playlists could be smaller than maxPlaylists
    print("Playlists: ", numPlaylists)
    print("Unique tracks: ", numUniqueTracks)
    print("MinSup: ", minSup, "->", numPlaylists * minSup, "playlists")

    # Final result global frequent itemset
    globalFreqTrackSet = dict()
    # Storing global itemset with support count
    globalTrackSetWithSup = defaultdict(int)

    currentLSet = getAboveMinSup(uniqueTracks, playlists, minSup, globalTrackSetWithSup)  # L1Itemset
    globalFreqTrackSet[1] = currentLSet
    print("Tracks above minSup:", len(currentLSet), "->", len(currentLSet) / numUniqueTracks * 100, "%")

    k = 2
    # Calculating frequent track set
    while currentLSet and k <= kMax:
        # Self-joining Lk
        print("getUnion for k =", k, "...")
        candidateSet = getUnion(currentLSet, k)

        print("Done!", len(candidateSet), "Candidates")

        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k - 1)
        # Scanning itemSet for counting support
        print("getAboveMinSup for k =", k, "...")
        currentLSet = getAboveMinSup(
            candidateSet, playlists, minSup, globalTrackSetWithSup)

        print("Done!", len(currentLSet), "Tracks above minSup")

        if k == 2:  # tracks  can be used for recommendation
            printInfo(playlists, uniqueTracks, currentLSet)

        # Storing frequent itemset
        globalFreqTrackSet[k] = currentLSet

        k += 1

    print("Calculating rules...")
    rules = associationRule(globalFreqTrackSet, globalTrackSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])
    print("Done!")
    print("rules: ", len(rules))
    return globalFreqTrackSet, rules


def aprioriAdvanced(maxPlaylists, minSup, minConf=0.5, kMax=2):
    # get for each playlist a set tracks above minSup
    l1TracksDict = getL1TracksDict(maxPlaylists, minSup)
    l1PlaylistsDict = getL1PlaylistsDict(maxPlaylists, minSup)
    print(len(l1PlaylistsDict))
    l2TrackSet = getNextTrackSet(l1PlaylistsDict, l1TracksDict, minSup, maxPlaylists, 2)
    print(len(l2TrackSet))


aprioriFromDB(50, 0.04)
print("*"*80)
aprioriAdvanced(50, 0.04)
