from helperMethods import natural_keys, makeCSVUnique, createIndexForCSV, checkParamItems, append, \
    normalize_name, saveDF2CSV, createDfsForDb, getDataFromJson, getJsonFiles, readDF_from_CSV, \
    getL1ItemSet2ValuesFromCSV, getSup
import pandas as pd


def csvForDb(maxFiles):
    """
    Creates the csv-files we used to fill our database. For each table in db we create one csv-file. The columns of
    the csv-file corresponds to the columns of the table
    :param maxFiles: Number of files that should be converted to csv
    :return: Nothing, but creates a csv-file on the folder defined in helperMethods
    """
    i = 1
    filenames = getJsonFiles()
    for filename in sorted(filenames, key=natural_keys):
        if not filename.startswith('mpd.slice.') and filename.endswith(".json") or i > maxFiles:
            break

        for df, name in zip(createDfsForDb(filename), ["playlists", "artists", "tracks", "albums", "pConT"]):
            saveDF2CSV(df, f'{name}_{maxFiles}.csv', mode='a', header=(i == 1))

        print(i, ", ", filename)  # print status
        i += 1

    makeCSVUnique(f'albums_{maxFiles}')
    makeCSVUnique('tracks_{maxFiles}')
    makeCSVUnique(f'artists_{maxFiles}')
    createIndexForCSV(f'pConT_{maxFiles}')


def csvItem2Values(maxFiles, keys, values='pid', minSup=1):
    """
    Creates csv-files from json-files that can be used as a dictionary for our apriori-algorithm.
    :param minSup:  Total number of playlists, the item must appear in. All entry's with len(values) < minSup will be
                    filtered. Could be used to save memory. By default = 1, so the csv-file can be used for many cases.
    :param maxFiles: Number of files that should be converted to csv
    :param keys:    Can be one or more items defined in helperMethods.validItems. For each of that item(s) a csv-file
                    will be created. By default = {'track_name', 'track_uri', 'artist_uri', 'album_uri', 'name'}, in
                    this case 5 files will be created.
    :param values:  Can be one or more items. For each value a csv-file will be created.
                    E.g len(keys) = 2, len(values) = 2 -> will generate 4 files
    :return:    Nothing, but creates a csv-file(s) on the folder defined in helperMethods.
                The file is of the form:
                    key1,{'value1', 'value2', ...}
                    key2{'value1', 'value2', ...}
                    ...
                E.g. keys={'track_uri', 'album_uri'}, values={'pid'} (or 'pid'), maxFiles = 1000
                file 1: track_uri2pids_1000.csv
                    spotify:track:004skCQeDn1iLntSom0rRr,"{'1294', '1923'}"
                    spotify:track:005CGalYNgMNZcvWMIFeK8,{'1487'}
                    ...
                file 2: album_uri2pids_1000.csv
                    spotify:album:00045VFusrXwCSietfmspc,"{'721228', '79257', '511709', '243234', '354319'}"
                    spotify:album:0005lpYtyKk9B3e0mWjdem,{'697783'}
                    ...
    """
    keys = checkParamItems("csvItem2Pids", keys)
    values = checkParamItems("csvItem2Values", values)
    formattedValues = set()
    for value in values:
        value = value.lower()
        formattedValues.add(value[-1] if value.endswith("s") else value)

    i = 1
    filenames = getJsonFiles()
    for filename in sorted(filenames, key=natural_keys):
        if not filename.startswith('mpd.slice.') and filename.endswith(".json") or i > maxFiles:
            break

        data = getDataFromJson(filename)
        df_pConT = pd.json_normalize(data['playlists'], record_path=['tracks'], meta=['pid', 'name'])
        df_pConT['name'] = df_pConT['name'].apply(normalize_name)
        df_pConT['track_name'] = df_pConT['track_name'] + " by " + df_pConT['artist_name']

        for item in keys:
            for value in values:
                df = df_pConT.groupby(item)[value].apply(set).reset_index(name=f'{value}s')
                saveDF2CSV(df, f'{item}2{value}s_{maxFiles}.csv', mode='a', header=(i == 1))

        print(i, ", ", filename)  # print status
        i += 1
    for item in keys:
        for value in values:
            df = readDF_from_CSV(f'{item}2{value}s_{maxFiles}.csv')
            df = df.groupby(item)[f'{value}s'].apply(append).reset_index(name=f'{value}s')
            oldLen = len(df[item])
            df = df[df[f'{value}s'].map(len) >= minSup]
            saveDF2CSV(df, f'{item}2{value}s_{maxFiles}.csv')
            print(item, "-> oldlen: ", oldLen, "newlen:", len(df[item]))


def csvPid2ItemsFromCSV(maxFiles, items):
    """
    This method is not very useful, because it is faster the generate the pid2ItemsDict with the method
    getL1Pid2ItemSetsFromDict() then to call getL1Pid2ItemSetsFromCSV() witch would use the file created by this
    method.
    Works like csvItem2Values but generates the csv-file from current csv-file instead of a json-file.
    For that reason csvItem2Values have to be called first with the parameters
    (maxFiles=maxFiles, keys=items, values='pid', minSup=whatever you want to work with. Suggested 1)
    :param maxFiles: Only needed to find and save the right file
    :param items: Set of items for which a csv-file should be created. Is also used to find the right file
    :return: Nothing, but creates a csv-file on the folder defined in helperMethods
                The file is of the form:
                    pid,"{'item1', 'item2', ...}"
                E.g. items = {'track_uri'}
                    295,"{'spotify:track:3NhbibAeeuBMArKoFYQ2Vd', 'spotify:track:4mKGSQxJk5mKOlMmMnW6gx', ...}"
                    ...
    """
    items = checkParamItems("csvPid2Items", items)
    for item in items:
        filename = f'{item}2Pids_{maxFiles}.csv'
        df = readDF_from_CSV(filename)
        print(filename)
        # get all unique pids
        pids = append(df['pids'])
        pid2Items = {pid: set() for pid in pids}

        for index, row in df.iterrows():
            for pid in row['pids'][1:-1].split(', '):
                pid2Items[pid.replace("'", "")].add(row[item])
        df_pid2Items = pd.DataFrame()
        df_pid2Items['pid'] = pid2Items.keys()
        df_pid2Items['items'] = pid2Items.values()
        saveDF2CSV(df_pid2Items, f'pid2{item}s_{maxFiles}.csv')


def savePopularTracks(maxFiles):
    """
    Before use, call
        - preprocessingToCSV.csvItem2Values(maxFiles=maxFiles, keys='track_uri', values='pid', minSup=1)
        - preprocessingToCSV.csvItem2Values(maxFiles=maxFiles,
                                            keys={'artist_uri', 'album_uri'},
                                            values='track_uri',
                                            minSup=1)

    Creates 3 csv-files with the most popular tracks
        - for each artist
        - for each album
        - at all
    Can be used to make predictions with artist- or album-rules or or when there aren't enough rules
    to predict 500 songs
    :param maxFiles: Only used to find the right file
    :return: Nothing. Creates 3 files in folder ../data_rules/.
    The files are of the form:
    album_uri2track_uris_sorted.csv:
        frozenset({'spotify:album:003sFH4G9RLE253AFIJ0YJ'}),"['spotify:track:5ahvjrjn7ymaeaWKFZrsca', ...]"
        frozenset({'spotify:album:004EYz2DQttcGvyTQGDmLp'}),"['spotify:track:7LxsehOTDXTpsuWAMWXe9o', ...]"
        ...
    artist_uri2track_uris_sorted.csv:
        frozenset({'spotify:artist:006Mv4bnAJGVnasH1pbDEO'}),"['spotify:track:56sreEUrAes3e5dFhTMm8S', ...]"
        frozenset({'spotify:artist:007pt2ONVI5ZWisox0DoP3'}),"['spotify:track:6O2cUoD35CeRQkIHhXmyuP', ...]"
        ...
    mostPopularTracks.csv:
        spotify:track:7KXjTSCq5nL1LoYtL7XAwS
        spotify:track:1xznGGDReH1oQq0xzbwXa3
        spotify:track:3a1lNhkSLSkpJE4MSHpDu9
        ...
    """
    for item in {'artist_uri', 'album_uri'}:
        item2track_uris = getL1ItemSet2ValuesFromCSV(item=item, value='track_uri', minSup=1, maxFiles=maxFiles)
        track_uri2Pids = getL1ItemSet2ValuesFromCSV(item='track_uri', value='pid', minSup=1, maxFiles=maxFiles)
        item2track_urisSorted = [[item, list(track_uris)]
                                 for item, track_uris in item2track_uris.items()]
        for row in item2track_urisSorted:
            row[1].sort(key=lambda x: len(getSup(x, track_uri2Pids)), reverse=True)
        saveDF2CSV(pd.DataFrame(item2track_urisSorted), f'{item}2track_uris_sorted.csv', path='../data_rules/')
    mostPopularTracks = [track_uri for track_uri in track_uri2Pids]
    mostPopularTracks.sort(key=lambda x: len(track_uri2Pids[x]), reverse=True)
    saveDF2CSV(pd.DataFrame(mostPopularTracks), "mostPopularTracks.csv", path='../data_rules/')


def createFiles():
    """
    Creates the most important csv-files for our current level of knowledge.
    :return: Nothing, only calls methods
    """
    # For testing
    csvItem2Values(2, 'name', {'track_uri', 'album_uri', 'artist_uri', 'pid'})
    csvItem2Values(2, {'track_uri', 'album_uri', 'artist_uri', 'name'}, 'pid')
    csvItem2Values(2, {'album_uri', 'artist_uri'}, 'track_uri')
    # For final prediction / rule-calculation
    csvItem2Values(1000, 'name', {'track_uri', 'album_uri', 'artist_uri', 'pid'})
    csvItem2Values(1000, {'track_uri', 'album_uri', 'artist_uri', 'name'}, 'pid')
    csvItem2Values(1000, {'album_uri', 'artist_uri'}, 'track_uri')
    savePopularTracks()



