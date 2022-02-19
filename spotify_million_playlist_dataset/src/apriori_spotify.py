import pandas as pd
from db import getL1ItemSet2Pids, getL1Pid2ItemSets, getNumUniqueItems
from helperMethods import getL1Pid2ItemSetsFromDict, getL1ItemSet2ValuesFromCSV, \
    checkParamItem, getSup, saveDF2CSV, normalize_uri, powerset
from progressBar import ProgressBar


def aprioriSpotify(item, maxPid, minSup=2, minConf=0.2, kMax=2, b=10, p=10, dbL1ItemSets=False):
    """

    :param item:    The item the apriori-algorithm creates rules for.
                    Could be either track_uri, track_name, artist_uri or album_uri
    :param maxPid:  Number of playlists that limits the load of data. Can be used for faster testing
    :param minSup:
    :param minConf:
    :param kMax:
    :param b:
    :param p:
    :param dbL1ItemSets:
    :return:
    """
    allRules = list()
    checkParamItem('aprioriSpotify', item)
    if dbL1ItemSets:    # get l1ItemSets from db
        numUniqueItems = getNumUniqueItems(item, maxPid)  # calculation may take longer than the ssh tunnel exists
        l1Pid2ItemSets = getL1Pid2ItemSets(item, maxPid, minSup)
        l1ItemSet2Pids = getL1ItemSet2Pids(item, maxPid, minSup)
    else:   # get l1ItemSets from csv-file
        maxFiles = (maxPid - 1) // 1000 + 1
        maxPid = maxFiles * 1000
        numUniqueItems = getNumUniqueItems(item, maxPid)  # calculation may take longer than the ssh tunnel exists
        l1ItemSet2Pids = getL1ItemSet2ValuesFromCSV(item=item, minSup=minSup, maxFiles=maxFiles)
        l1Pid2ItemSets = getL1Pid2ItemSetsFromDict(l1ItemSet2Pids)  # this one is faster

    print(f"k: 1 -> {len(l1ItemSet2Pids)} items {round(len(l1ItemSet2Pids) / numUniqueItems * 100, 2)}%)")
    currentItemSet2Pids = l1ItemSet2Pids
    currentPid2ItemSets = l1Pid2ItemSets

    for k in range(2, kMax + 1):
        currentItemSet2Pids, currentPid2ItemSets = getNextItemSets(currentItemSet2Pids, currentPid2ItemSets,
                                                                   minSup, k, b, p)
        print("k: ", k, "->", len(currentItemSet2Pids), "itemSets")
        allRules.extend(associationRules(l1ItemSet2Pids, currentItemSet2Pids, minConf))

    saveAndSortRules(allRules, f'{maxFiles}_{item}2{item}.csv')


def aprioriPname(consequents, maxPid, minSup=2, minConf=0.2):
    # rules of form [[frozenset({pname}), frozenset({item}), confidence, supPname], ...]
    maxFiles = (maxPid - 1) // 1000 + 1
    rules = list()
    checkParamItem('aprioriPname', consequents)
    pname2Pids = getL1ItemSet2ValuesFromCSV(item='name', value='pids', minSup=minSup, maxFiles=maxFiles)
    pname2Items = getL1ItemSet2ValuesFromCSV(item='name', value=consequents, minSup=1, maxFiles=maxFiles)
    item2Pids = getL1ItemSet2ValuesFromCSV(item=consequents, value='pids', minSup=1, maxFiles=maxFiles)
    countProgress = 0
    pb = ProgressBar(total=len(pname2Pids), prefix='Calculate rules', timing=True)
    for pname in pname2Pids:
        pb.printProgressBar(countProgress)
        pidsPname = pname2Pids[pname]
        for item in pname2Items[pname]:
            confidence = len(getSup(item, item2Pids).intersection(pidsPname)) / len(pidsPname)
            if confidence > minConf:
                rules.append([pname, item, confidence, len(pidsPname)])
        countProgress += 1

    saveAndSortRules(rules, f'{maxFiles}_name2{consequents}.csv')


def getNextItemSets(currItemSet2Pids, currPid2ItemSets, minSup, k, b, p):
    """"
    Combines apriori_first.getAboveMinSup() and apriori_first.getUnion() and is also way faster for k = 2.
    :param currItemSet2Pids:    This parameter is used to iterate over the playlists of a itemSet to
                                concatenate it only with itemSets from playlists itself appears in.
                                Due to an average occurrence of 10 playlists per track it reduces the number of
                                candidates significantly
                                This parameter is also used to get the support of a itemSet by counting the playlists

    :param currPid2ItemSets:    This parameter is necessary to get the itemSets above minSup of a playlist, so that
                                 we don't have to iterate over the entire playlist
    :param minSup:   Specifies the minimum number of playlists the itemSet must appear in
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
                {
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
                {
                    frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1',
                    'spotify:track:3H1LCvO3fVsK2HPguhbml0'}): {0, 5, ...},
                    frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4',
                    'spotify:track:3H1LCvO3fVsK2HPguhbml0'}): {0, 5, ...},
                    ...
                }
    """
    nextItemSet2Pids = dict()
    countProgress = 0
    pb = ProgressBar(total=len(currItemSet2Pids), prefix=f'Calculate l{k}ItemSet2Pids', timing=True)
    for itemSet1 in currItemSet2Pids:  # for each k-1 itemSet above minSup
        pb.printProgressBar(countProgress)
        proveCandidates(itemSet1, b, k, p, minSup, currItemSet2Pids, currPid2ItemSets, nextItemSet2Pids)
        countProgress += 1
    # for better performance in next round we need a dict that gives us for each playlist the k-itemSets above minSup
    nextPid2ItemSets = getNextPid2ItemSets(currPid2ItemSets, nextItemSet2Pids)

    return nextItemSet2Pids, nextPid2ItemSets


def proveCandidates(itemSet1, b, k, p, minSup, currItemSet2Pids, currPid2ItemSets, nextItemSet2Pids):
    candidates = list()
    iS1Pids = currItemSet2Pids[itemSet1]
    for iS1Pid in iS1Pids:  # for each playlist the k-1 itemSet appears in
        for itemSet2 in currPid2ItemSets[iS1Pid]:  # for each k-1 itemSet above minSup in playlist
            intersectionPids = iS1Pids.intersection(currItemSet2Pids[itemSet2])
            unionSet = frozenset(itemSet1.union(itemSet2))
            # itemSet1 & itemSet2 must be together in min. minSup playlists
            if len(intersectionPids) >= minSup and len(unionSet) == k:
                candidates.append((unionSet, intersectionPids))
        p -= 1
        if p == 0:  # only iterate over max p playlists
            break
    # on this point the candidate is proved for minSup and length so it can be used for rules
    # only save the best b candidates with highest support
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    for i in range(0, b + 1 if b < len(candidates) and b != -1 else len(candidates)):
        nextItemSet2Pids[candidates[i][0]] = candidates[i][1]
    candidates.clear()


def getNextPid2ItemSets(currPid2ItemSets, nextItemSet2Pids):
    nextPid2ItemSets = dict()
    for pid in currPid2ItemSets:
        nextPid2ItemSets[pid] = set()

    for itemSet in nextItemSet2Pids:
        for pid in nextItemSet2Pids[itemSet]:
            nextPid2ItemSets[pid].add(itemSet)

    return nextPid2ItemSets


def associationRules(l1ItemSet2Pids, currItemSet2Pids, minConf):
    # rules of form [[frozenset({antecedents}), frozenset({consequents}), confidence, supAntecedent], ...]
    rules = list()
    countProgress = 0
    pb = ProgressBar(total=len(currItemSet2Pids), prefix='Calculate rules', timing=True)
    for itemSet, pids in currItemSet2Pids.items():
        pb.printProgressBar(countProgress)
        for antecedent in frozenset(powerset(itemSet)):
            supAntecedent = len(getSup(antecedent, l1ItemSet2Pids))
            confidence = len(currItemSet2Pids[itemSet]) / supAntecedent
            consequents = itemSet.difference(antecedent)
            if confidence > minConf:
                rules.append([antecedent, consequents, confidence, supAntecedent])
        countProgress += 1
    return rules


def saveAndSortRules(rules, filename):
    # sort for antecedent alphabetically and sort rules for these antecedent by confidence
    rules.sort(key=lambda x: (x[0], x[2]), reverse=True)
    dfRules = pd.DataFrame(rules, columns=['antecedent', 'consequent', 'confidence', 'supAntecedent'])
    dfRules['antecedent'] = dfRules['antecedent'].apply(normalize_uri)
    dfRules['consequent'] = dfRules['consequent'].apply(normalize_uri)
    saveDF2CSV(dfRules, filename=filename, path='../data_rules/')

    # sort for antecedentSup and confidence
    rules.sort(key=lambda x: (x[3], x[2]), reverse=True)
    antecedents = set()
    for rule in rules:
        antecedents.add(rule[0])
    print("Top rules: ", rules[0:500])
    print("Total rules: ", len(rules))
    print("Total antecedents: ", len(antecedents))


# aprioriSpotify('track_uri', maxPid=2000, minSup=5)
# aprioriPname('artist_uri', 1000000, minConf=0.2)
aprioriSpotify('track_uri', 2000, p=-1, minSup=10, dbL1ItemSets=False)
