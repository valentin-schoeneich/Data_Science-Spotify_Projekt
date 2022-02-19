from collections import defaultdict
from itertools import combinations
from db import getFromDB
from helperMethods import powerset

'''
Our first attempt to create rules with apriori-algorithm.
We used a existing repository from https://github.com/chonyy/apriori_python and rebuild it, that it works 
for our data.
'''


def getAboveMinSup(candidateTrackSets, playlistSets, minSup, globalTrackSetWithSup):
    """
    This method filters the track-sets by his support.  The candidates are generated from getUnion()
    :param candidateTrackSets: Set of track-frozensets with support above and under minSup
    :param playlistSets:    List of sets. Each set represents a playlist with unique tracks
    :param minSup:  Likelihood of a track being included in a playlist
                    e.g. for minSup = 0.5 this function returns all tracks which are included in 50% of all playlists
    :param globalTrackSetWithSup: Dictionary that maps a track-set to his support
    :return: All track-sets above mindSup in form of the input parameter candidateTrackSets
    """
    freqTrackSet = set()
    localTrackSetWithSup = defaultdict(int)
    i = 0
    for track in candidateTrackSets:
        i += 1
        for playlistSet in playlistSets:
            if track.issubset(playlistSet):
                globalTrackSetWithSup[track] += 1
                localTrackSetWithSup[track] += 1
    for track, supCount in localTrackSetWithSup.items():
        support = float(supCount / len(playlistSets))
        if support >= minSup:
            freqTrackSet.add(track)
    return freqTrackSet


def getUnion(trackSet, length):
    """
    This method was used by our first apriori-attempt. It is too slow for more than 10.000 tracks
    and returns way too much candidates in first round from l1TrackSet to l2TrackSet.
    With length = 2 it returns len(trackSet) * (len(trackSet)-1) / 2 candidates.
    :param trackSet:    A set of all frequent track-sets above minSup in form of {frozenset({'track_uri'}), ...}
    :param length:      The length that the union track-set should have
    :return:    Returns a set of sub-set's just like the parameter trackSet but with the new length for each sub-set
                E.g trackSet = {A,B,C,D}, length = 2
                returns: {{A,B}, {A,C}, {A,D}, {B,C}, {B,D}, {C,D}}
    """
    return set([i.union(j) for i in trackSet for j in trackSet if len(i.union(j)) == length])


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


def associationRuleUnchanged(freqItemSet, itemSetWithSup, minConf):
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


def printInfo(unique_tracks, trackSet):
    usableTracks = list()
    for tracks in trackSet:
        for track in tracks:
            usableTracks.append(track)
    usableTracks = set(usableTracks)
    num_unique_tracks = len(unique_tracks)
    num_usableTracks = len(usableTracks)

    print("tracks for recom.: ", num_usableTracks, "->", round(num_usableTracks / num_unique_tracks * 100, 2), "%")


def aprioriFromDB(maxPlaylists, minSup, minConf=0.5, kMax=2):
    """
    Main method of this file.
    :param maxPlaylists: Defines the load of data requested from our database
    :param minSup:  Percentage of minimum playlists a track have to appear in, that he is used for rule-calculation
    :param minConf: Rates the rule. If the rule A -> B has the confidence 0.5, it means that B appears in 50% of
                    playlists, A appears in
    :param kMax:    Limits the iterations of this method by the size of the track-sets.
                    If kMax = 2, a rule could look like:
                    [{'spotify:track:3H1LCvO3fVsK2HPguhbml0'}, {'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}, 1.0]
                    If kMax = 3, it also could look like:
                    [{'spotify:track:3H1LCvO3fVsK2HPguhbml0', 'spotify:track:3H1LCvO3fVsKabdshbmxy'},
                    {'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}, 1.0]
    :return:    Rules and the globalFreqTrackSet
    """
    uniqueTracks, playlists = getFromDB(maxPlaylists)
    numUniqueTracks = len(uniqueTracks)
    numPlaylists = len(playlists)  # number of playlists could be smaller than maxPlaylists
    print("*" * 80, "\nPlaylists: ", numPlaylists)
    print("Unique tracks: ", numUniqueTracks)
    print("MinSup: ", minSup, "->", numPlaylists * minSup, "playlists")

    # Final result global frequent itemset
    globalFreqTrackSet = dict()
    # Storing global itemset with support count
    globalTrackSetWithSup = defaultdict(int)

    currentLSet = getAboveMinSup(uniqueTracks, playlists, minSup, globalTrackSetWithSup)  # L1Itemset
    globalFreqTrackSet[1] = currentLSet
    print("Tracks above minSup:", len(currentLSet), "->", len(currentLSet) / numUniqueTracks * 100, "%")
    print("*" * 80)
    k = 2
    # Calculating frequent track set
    while currentLSet and k <= kMax:
        # Self-joining Lk
        print(f"getUnion for k ={k} ...", end='')
        candidateSet = getUnion(currentLSet, k)
        print(" -> Done!", len(candidateSet), "Candidates")

        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k - 1)
        # Scanning itemSet for counting support
        print(f"getAboveMinSup for k = {k} ...", end='')
        currentLSet = getAboveMinSup(
            candidateSet, playlists, minSup, globalTrackSetWithSup)
        print(" -> Done!", len(currentLSet), "Tracks above minSup")

        if k == 2:  # tracks  can be used for recommendation
            printInfo(uniqueTracks, currentLSet)

        # Storing frequent itemset
        globalFreqTrackSet[k] = currentLSet

        k += 1

    print("Calculating rules...", end='')
    rules = associationRuleUnchanged(globalFreqTrackSet, globalTrackSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])
    print(" -> Done!")
    print("Total rules: ", len(rules))
    return globalFreqTrackSet, rules


x, y = aprioriFromDB(10, 0.2)
print(y)
