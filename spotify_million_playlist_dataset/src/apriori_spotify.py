import itertools
import time
from collections import defaultdict
from itertools import chain, combinations

from db import *
from progressBar import *


def getAboveMinSup(unique_tracks, playlists, minSup, globalTrackSetWithSup):
    """
    :param unique_tracks:
    :param playlists:
    :param minSup:  Likelihood of a track being included in a playlist
                    e.g. for minSup = 0.5 this function returns all tracks which are included in a playlist
                    with a likelihood of 50%
    :param globalTrackSetWithSup:
    :return:
    """
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


'''
***************************************************************
Methods for new approach
***************************************************************
'''


def getNextTracksDict(currPid2ItemSets, nextItemSet2Pids):
    nextPid2ItemSets = dict()
    for pid in currPid2ItemSets:
        nextPid2ItemSets[pid] = set()

    for itemSet in nextItemSet2Pids:
        for pid in nextItemSet2Pids[itemSet]:
            nextPid2ItemSets[pid].add(itemSet)

    return nextPid2ItemSets


def getNextItemSets(currItemSet2Pids, currPid2ItemSets, minSupPercent, maxPid, k, b):
    """"
    Combines getAboveMinSup() and getUnion() and is also way faster for k = 2.
    :param currItemSet2Pids:    This parameter is used to iterate over the playlists of a itemSet to
                                concatenate it only with itemSets from playlists itself appears in.
                                Due to an average occurrence of 10 playlists per track it reduces the number of
                                candidates significantly
                                This parameter is also used to get the support of a itemSet by counting the playlists

    :param currPid2ItemSets:    This parameter is necessary to get the itemSets above minSup of a playlist, so that
                                 we don't have to iterate over the entire playlist
    :param minSupPercent:   Specifies the minimum number of playlists the itemSet must appear in
    :param maxPid:          Number of playlists that limits the load of data. Can be used for faster testing.
    :param k:               Size of itemSet
    :param b:               Number of b - nextItemSets with highest support that should be saved
                            Standard value of b is -1. If b is -1 all itemSets above minSUp with length k will be saved.
                            With b != -1 the program isn't deterministic
    :return:    pid2Itemsets and itemSet2Pids for k-itemSets in form of:
                pid2Itemsets:
                {
                    pid: {frozenset({'item', 'item', ...}), ...}
                    ...
                }
                e.g. with item = track_uri and k = 2:
                    5: {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1', 'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}),
                        frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4', 'spotify:track:3H1LCvO3fVsK2HPguhbml0'})
                        ...},
                    0: {frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0', 'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}),
                        frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4', 'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'})
                        ...},
                    ...
                }
                itemSet2Pids
                {
                    frozenset({'item', 'item', ...}): {pid, pid, ...},
                    ...
                }
                e.g. with item = track_uri and k = 2:
                    frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1',
                    'spotify:track:3H1LCvO3fVsK2HPguhbml0'}): {0, 5, ...},
                    frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4',
                    'spotify:track:3H1LCvO3fVsK2HPguhbml0'}): {0, 5, ...},
                    ...
                }
    """
    minSup = maxPid * minSupPercent  # the actual number of playlists the item must appears in
    nextItemSet2Pids = dict()
    countProgress = 0
    pb = ProgressBar(total=len(currItemSet2Pids), prefix='Calculate nextItemSet2Pids', timing=True)
    for itemSet1 in currItemSet2Pids:  # for each k-1 itemSet above minSup
        pb.printProgressBar(countProgress)
        proveCandidates(itemSet1, b, k, minSup, currItemSet2Pids, currPid2ItemSets, nextItemSet2Pids)
        countProgress += 1
    # for better performance in next round we need a dict that gives us for each playlist the k-itemSets above minSup
    nextPid2ItemSets = getNextTracksDict(currPid2ItemSets, nextItemSet2Pids)

    return nextItemSet2Pids, nextPid2ItemSets


def proveCandidates(itemSet1, b, k, minSup, currItemSet2Pids, currPid2ItemSets, nextItemSet2Pids):
    candidates = [
        (frozenset(itemSet1.union(itemSet2)), currItemSet2Pids[itemSet1].intersection(currItemSet2Pids[itemSet2]))
        for iS1Pid in currItemSet2Pids[itemSet1]  # for each playlist the k-1 itemSet appears in
        for itemSet2 in currPid2ItemSets[iS1Pid]  # for each k-1 itemSet above minSup in playlist
        # itemSet1 & itemSet2 must be together in min. minSup playlists
        if len(currItemSet2Pids[itemSet1].intersection(currItemSet2Pids[itemSet2])) >= minSup
           and len(itemSet1.union(itemSet2)) == k]
    # on this point the candidate is proved for minSup and length so it can be used for rules
    # only save the best b candidates with highest support
    candidates.sort(key=takeSupport, reverse=True)
    for i in range(0, b + 1 if b < len(candidates) and b != -1 else len(candidates)):
        nextItemSet2Pids[candidates[i][0]] = candidates[i][1]
    candidates.clear()


def takeSupport(elem):
    return len(elem[1])


def aprioriAdvanced(item, maxPid, minSup, minConf=0.5, kMax=2, b=-1):
    # get for each playlist a set of items above minSup
    currentPid2ItemSets = getL1Pid2ItemSets(item, maxPid, minSup)
    # get for each item a list of playlists it appears in
    currentItemSet2Pids = getL1ItemSet2Pids(item, maxPid, minSup)
    print(f"k: 1 -> {len(currentItemSet2Pids)} items "
          f"({round(len(currentItemSet2Pids) / getNumUniqueItems(item, maxPid) * 100, 2)}%)")
    for k in range(2, kMax + 1):
        currentItemSet2Pids, currentPid2ItemSets = getNextItemSets(currentItemSet2Pids, currentPid2ItemSets, minSup,
                                                                   maxPid, k, b)
        print("k: ", k, "->", len(currentItemSet2Pids), "itemSets")
    sup = 0

    items = set()
    for key in currentItemSet2Pids:
        sup += len(currentItemSet2Pids[key])
        for i in key:
            items.add(i)

    print("AVG Supply:", round(sup / max(1, len(currentItemSet2Pids)), 2))
    print(f"({round(len(items) / getNumUniqueItems(item, maxPid) * 100, 2)}%)")

    # print(currentItemSet2Pids)


# aprioriFromDB(100, 0.02, kMax=3)
# aprioriFromDB(200, 0.01)
print("*" * 80)
aprioriAdvanced('track_uri', 40000, 0.00005, b=5)
