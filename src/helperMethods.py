import re
import pandas as pd
from progressBar import *
import os
import json
from itertools import chain, combinations

validItems = {'track_name', 'track_uri', 'artist_uri', 'album_uri', 'name', 'pid'}
pathToProcessedData = '../data_processed/'
pathToData = '../data/'

'''
*************************************
Helper-methods for preprocessing
*************************************
'''


def createDfsForDb(filename):
    """
    Outsourcing from preprocessingToCSV.csvForDb.
    :param filename: The filename of the current json-file
    :return: A list of dataframes that represents the tables of our database
    """
    data = getDataFromJson(filename)
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

    return [df_playlists, df_artist, df_tracks, df_album, df_pConT]


def getDataFromJson(filename, path=pathToData):
    """
    :param path:
    :param filename: The filename of the current json-file
    :return: The data from the json-file in form of a dataframe
    """
    f = open(os.sep.join((path, filename)))
    js = f.read()
    f.close()
    data = json.loads(js)
    return data


def getJsonFiles():
    """
    :return: Returns a list of files in the folder defined by pathToData
    """
    return os.listdir(pathToData)


def makeCSVUnique(filename):
    """
    Outsourcing from preprocessingToCSV.csvForDb. Drops duplicates in a specified csv-file.
    :param filename: filename of the csv-file without .csv-suffix and path.
    :return: Nothing, only edits and saves the specified csv-file
    """
    print('dropping duplicates in ' + filename)
    df = pd.read_csv(f'{pathToProcessedData}{filename}.csv')
    df = df.drop_duplicates()
    df.to_csv(f'{pathToProcessedData}{filename}.csv', index=False)


'''
*************************************
General helper-methods
*************************************
'''


def appendSet(xs):
    """
    Convert a set in form of a string or a set of subsets (in form of a string) to a real set.
    :param xs: E.g. '{'x', 'y', 'z'}' or {'{'x', 'y', 'z'}', '{'1', '2', '3'}'}
    :return: Returns a set of elements given by xs
    """
    if type(xs) == list or type(xs) == set or type(xs) == pd.core.series.Series:
        return {elem.replace("'", "") for x in xs for elem in x[1:-1].split(', ')}
    else:
        return {elem.replace("'", "") for elem in str(xs)[1:-1].split(', ')}


def appendList(xs):
    """
    Convert a iterable in form of a string or a set of subsets (in form of a string) to a list.
    :param xs: E.g. '{'x', 'y', 'z'}' or {'{'x', 'y', 'z'}', '{'1', '2', '3'}'}
    :return: Returns a list of elements given by xs
    """
    if type(xs) == list or type(xs) == set or type(xs) == pd.core.series.Series:
        return [elem.replace("'", "") for x in xs for elem in x[1:-1].split(', ')]
    else:
        return [elem.replace("'", "") for elem in str(xs)[1:-1].split(', ')]


def atoi(filename):
    return int(filename) if filename.isdigit() else filename


def natural_keys(filename):
    """
    Used to sort files alphanumerical
    :param filename: Name of the file
    :return: Returns a list
    """
    return [atoi(c) for c in re.split(r'(\d+)', filename)]


def normalize_name(name):
    """
    Copied from stats.py. Normalize the name of a playlist
    :param name: name of playlist
    :return: normalized name of playlist
    """
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.replace("'", "")
    return name


def normalize_uri(uri):
    """
    Returns the pure uri.
    :param uri: E.g. "frozenset({'track_uri'})"
    :return: E.g. track_uri
    """
    uri = str(uri)
    uri = uri.replace("frozenset", "")
    return re.sub(r"[\"'{}()]", "", uri).strip()


def checkParamItems(name, items, valid=validItems):
    """
    This method is used to check the passed parameter of a method
    :param valid:   All items that should be accepted
    :param name:    Name of the method calling this method
    :param items:   Items which the calling method has get as a parameter
    :return:    Items converted to a set
    """
    if type(items) == list:
        items = {x for x in items}
    elif type(items) != set:
        items = {items}
    if len(items.union(valid)) > len(valid):
        raise ValueError(f'{name}: item must be one of {valid}')
    return items


def checkParamItem(name, item, valid=validItems):
    """
    This method is used to check the passed parameter of a method. Works like checkParamItems, but only for one item.
    :param valid:   All items that should be accepted
    :param name:    Name of the method calling this method
    :param item:   Item which the calling method has get as an parameter
    :return:    Nothing, raising error only
    """
    if item not in valid:
        raise ValueError(f'{name}: item must be one of {valid}')


def createIndexForCSV(filename):
    """
    Creates a column of indexes for each row.
    :param filename: Name of the file.
    :return: Nothing, only edits and saves the file
    """
    df = pd.read_csv(f'{pathToProcessedData}{filename}.csv')
    df.to_csv(f'{pathToProcessedData}{filename}.csv')


def saveDF2CSV(df, filename, mode='w', index=False, header=True, path=pathToProcessedData):
    df.to_csv(f'{path}{filename}', mode=mode, index=index, header=header)


def readDF_from_CSV(filename, path=pathToProcessedData):
    return pd.read_csv(f'{path}{filename}')


def powerset(s):
    """
    Copied from https://github.com/chonyy/apriori_python
    :param s: A itemSet. E.g. {a, b, c}
    :return:  A set of subsets with length < s. E.g. {{a}, {b}, {c}, {a,b}, {a,c}, {b,c}}
    """
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


'''
*************************************
Helper-methods for apriori
*************************************
'''


def getL1ItemSet2ValuesFromCSV(item, value='pid', minSup=2, maxFiles=1000):
    """
    Before use, call preprocessingToCSV.csvItem2Values() with parameters:
    (maxFiles=maxFiles, keys=item, values=value, minSup=minSup or smaller):
    :param item: Will be the key of the dictionary
    :param value:   The value of the dictionary
    :param minSup:  Minimum Support as total number of playlists
    :param maxFiles: Can be used to load a smaller file
    :return: A dictionary that lists for each item all values. It is of the form:
                {
                    frozenset({'item'}): {'value', 'value', ...},
                    ...
                }
                e.g. with item = track_uri and value='pid'
                {
                    frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'}): {'0', '5', ...},
                    frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}): {'0', '5', ...},
                    ...
                }
    """
    value = value.lower()
    valueS = value if value.endswith("s") else f'{value}s'
    try:
        filename = f'{item}2{valueS}_{maxFiles}.csv'
        print(f'getL1ItemSet2ValuesFromCSV:\n\t- Load {filename}...', end='')
        df = readDF_from_CSV(filename)
    except:
        filename = f'{item}2{valueS}_1000.csv'
        print(f'\r\t- Load {filename}...', end='')
        df = readDF_from_CSV(filename)

    print(f" -> Done!\n\t- Convert {valueS} ...", end='')
    df[valueS] = df[valueS].apply(lambda x: {y for y in appendSet(x)})
    print(" -> Done!\n\t- Filter items above minSup...", end='')
    df = df.loc[df[valueS].map(len) >= minSup]
    print(" -> Done!\n\t- Convert item to frozenset...", end='')
    df[item] = df[item].apply(lambda x: frozenset([x]))
    print(" -> Done!")
    return {row[item]: row[valueS] for index, row in df.iterrows()}


def getL1Pid2ItemSetsFromCSV(item, maxFiles=1000):
    """
    Slower then getL1Pid2ItemSetsFromDict()
    :param item: Will be the value of the dictionary
    :param maxFiles: Can be used to load a smaller file
    :return: A dictionary that lists for each pid all items. It is of the form:
                {
                    pid: {frozenset({'item', 'item', ...}), ...}
                    ...
                }
                e.g. with item = track_uri
                {
                    5: {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'}),
                        frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4'})
                        ...},
                    0: {frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0'}),
                        frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4'})
                        ...},
                    ...
                }
    """
    try:
        filename = f'pid2{item}s_{maxFiles}.csv'
        print(f'getL1Pid2ItemSetsFromCSV:\n\t- Load {filename}...', end='')
        df = readDF_from_CSV(filename)
    except:
        filename = f'pid2{item}s_1000.csv'
        print(f'\r\t- Load {filename}...', end='')
        df = readDF_from_CSV(filename)
    print(" -> Done!\n\t- Convert items to frozensets...", end='')
    df['items'] = df['items'].apply(lambda x: {frozenset([y]) for y in appendSet(x)})
    print(" -> Done!\n\t- Convert pid to int...", end='')
    df['pid'] = df['pid'].apply(lambda x: int(x[1:-1]))
    print(" -> Done!")
    return {row['pid']: row['items'] for index, row in df.iterrows()}


def getL1Pid2ItemSetsFromDict(l1ItemSet2Pids):
    """
    Returns the same like getL1Pid2ItemSetsFromCSV() but instead of generating the dict by a csv-file, the csv-file
    will be created by a dictionary l1ItemSet2Pids
    :param l1ItemSet2Pids: Could be created by getL1ItemSet2ValuesFromCSV() or db.getL1ItemSet2Pids().
    :return: Defined in getL1Pid2ItemSetsFromCSV()
    """
    pb = ProgressBar(total=len(l1ItemSet2Pids) * 2, prefix='Calculate l1Pid2ItemSet from dict', timing=True)
    countProgress = 0
    allPids = set()
    for itemSet, pids in l1ItemSet2Pids.items():
        countProgress += 1
        pb.printProgressBar(countProgress)
        for pid in pids:
            allPids.add(pid)

    l1Pid2ItemSets = {pid: set() for pid in allPids}

    for itemSet, pids in l1ItemSet2Pids.items():
        countProgress += 1
        pb.printProgressBar(countProgress)
        for pid in pids:
            l1Pid2ItemSets[pid].add(itemSet)

    return l1Pid2ItemSets


def getSup(itemSet, l1ItemSet2Pids):
    """
    Returns the support of a given itemSet in form of a set with all pids it appears in
    :param itemSet: The itemSet for which to calculate the support
    :param l1ItemSet2Pids:  The dictionary where the support of single items can be read
    :return:  A set of pids e.g. {'2', '3', '42'}
    """
    if type(itemSet) not in {list, frozenset, set}:
        return l1ItemSet2Pids[frozenset([normalize_uri(itemSet)])]
    else:
        if len(itemSet) == 1:   # iterable with only one element e.g. [['x']]
            return getSup(itemSet.pop(), l1ItemSet2Pids)
        else:
            return _getSupFromIterable(itemSet, l1ItemSet2Pids)


def _getSupFromIterable(itemSet, l1ItemSet2Pids):
    """
    Splits an iterable recursively into its elements and returns the support from all elements
    :param itemSet: The itemSet for which to calculate the support
    :param l1ItemSet2Pids:  The dictionary where the support of single items can be read
    :return:  A set of pids e.g. {'2', '3', '42'}
    """
    if type(itemSet) not in {list, frozenset, set}:
        raise TypeError("getSupForIterable: type(itemSet) must be one of {list, frozenset, set}")
    if len(itemSet) < 2:
        raise ValueError("getSupForIterable: len(itemSet) should be greater than 1. Use getSup() instead.")
    intersectionPids = set()
    i = 0
    for item in itemSet:
        if i == 0:
            intersectionPids = getSup(item, l1ItemSet2Pids)
        else:
            intersectionPids = intersectionPids.intersection(getSup(item, l1ItemSet2Pids))
        i = 1
    return intersectionPids


def printSupInfo(item):
    """
    Prints infos about the support of an item which can be used for analysis
    :param item: The items for which the support should be counted. Defined in helperMethod.validItems
    :return: Nothing, print only
    """
    checkParamItem("printSupInfo", item)

    item2NumUnique = {'track_uri': 2262292, 'track_name': 2189699,
                      'artist_uri': 295860, 'artist_name': 287739,
                      'album_uri': 734684, 'album_name': 571627,
                      'name': 17379}
    df = readDF_from_CSV(f'{item}2Pids_1000.csv')
    df['support'] = df['pids'].apply(lambda pids: len(pids[1:-1].split(', ')))
    support = df.groupby('support')[item].apply(lambda x: len(x)).reset_index(name='countSup')
    # insert row on first line because items with support = 1 wasn't considered by csvItems2Pid to safe memory
    support.loc[-1] = [1, item2NumUnique[item] - len(df[item])]
    support.index = support.index + 1
    support = support.sort_index()
    support['mulSup'] = support['support'] * support['countSup']
    support['percentage'] = support['countSup'] / item2NumUnique[item] * 100
    support['percentageCumulative'] = support['percentage'].cumsum()
    print(support)
    print("Support mean: ", support['mulSup'].sum() / support['countSup'].sum())

