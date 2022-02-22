from helperMethods import readDF_from_CSV, getDataFromJson, normalize_name, appendList, normalize_uri
import statistics


def getRuleDict(filename):
    print(f"Load {filename}...", end='')
    df = readDF_from_CSV(filename, path='../data_rules/')
    df['rule'] = df[['consequent', 'confidence']].values.tolist()
    df_new = df.groupby('antecedent')['rule'].apply(list).reset_index(name='rules')
    print(" -> Done!")
    return {row['antecedent']: row['rules'] for index, row in df_new.iterrows()}


def getRuleDicts(item):
    pname2item = getRuleDict(f"1000_name2{item}.csv")
    item2item = getRuleDict(f"1000_{item}2{item}.csv")
    return pname2item, item2item


def getMostPopular(filename):
    print(f"Load {filename}...", end='')
    df = readDF_from_CSV(filename, path='../data_rules/')
    if len(df.columns) == 1:
        return df['0'].tolist()

    df['0'] = df['0'].apply(lambda x: normalize_uri(x))
    df['1'] = df['1'].apply(lambda x: appendList(x))
    print(" -> Done!")
    return {row['0']: row['1'] for index, row in df.iterrows()}


def tryPredict(dict, key):
    try:
        return {consequent[0] for consequent in dict[key]}
    except:
        return {}


def predForTrack(tracks, dictItem, itemName):
    predictions = set()
    for track in tracks:
        predictions.update(tryPredict(dictItem, track[itemName]))
    return predictions


def predForItem(items, dictItem):
    predictions = set()
    for item in items:
        predictions.update(tryPredict(dictItem, item))
    return predictions


def addPopular(trackPred, pred, popDict, maxTracks):
    for item in pred:
        i = 0
        for track in popDict[item]:
            if i > maxTracks:
                break
            i += 1
            trackPred.add(track)


def recommendation():
    submission = list()
    mpd_slice = getDataFromJson(filename="challenge_set.json", path='../')
    album2mostPopular = getMostPopular("album_uri2track_uris_sorted.csv")
    artist2mostPopular = getMostPopular("artist_uri2track_uris_sorted.csv")
    mostPopularTracks = getMostPopular("mostPopularTracks.csv")
    pname2track, track2track = getRuleDicts('track_uri')
    pname2artist, artist2artist = getRuleDicts('artist_uri')
    pname2album, album2album = getRuleDicts('album_uri')

    incomplete = 0
    count = 0
    x1 = 0
    x2 = 0
    lenPredictions = list()
    for playlist in mpd_slice['playlists']:
        print(count)
        szenario = checkSzenario(playlist)
        submission.append(playlist['pid'])
        trackPred, albumPred, artistPred = set(), set(), set()
        tracks = playlist['tracks']
        track_uris = set()
        for track in tracks:
            track_uris.add(track['track_uri'])

        if szenario != 4 and szenario != 6:  # playlist-name given
            pname = normalize_name(playlist['name'])
            # make predictions from playlist-name
            albumPred.update(tryPredict(pname2album, pname))
            artistPred.update(tryPredict(pname2artist, pname))
            trackPred.update(tryPredict(pname2album, pname))
        # make predictions from tracks
        albumPred.update(predForTrack(tracks, album2album, 'album_uri'))
        artistPred.update(predForTrack(tracks, artist2artist, 'artist_uri'))
        trackPred.update(predForTrack(tracks, track2track, 'track_uri'))

        trackPred = trackPred.difference(track_uris)

        if len(trackPred) < 5:
            print("Found", len(trackPred), "predictions for playlist:", playlist)
            submission.append(mostPopularTracks[0:500])
            incomplete += 1

        for i in range(0, 4):
            kmax = (500 - len(trackPred)) // max(len(albumPred) + len(artistPred), 1)
            addPopular(trackPred, albumPred, album2mostPopular, kmax)
            addPopular(trackPred, artistPred, artist2mostPopular, kmax)
            trackPred = trackPred.difference(track_uris)
            if len(trackPred) >= 500:
                x1 += 1
                break

            albumPred.update(predForItem(albumPred, album2album))
            artistPred.update(predForItem(artistPred, artist2artist))
            trackPred.update(predForItem(trackPred, track2track))
            trackPred = trackPred.difference(track_uris)
            if len(trackPred) >= 500:
                x2 += 1
                break

        if len(trackPred) < 500:
            incomplete += 1
        i = 0
        while len(trackPred) < 500:
            trackPred.add(mostPopularTracks[i])
            i += 1

        submission.append(trackPred)

        count += 1
        lenPredictions.append(len(trackPred))

    print(f'{incomplete} incomplete from {count}')
    print("Avg: ", statistics.mean(lenPredictions))
    print("max: ", max(lenPredictions))
    print("x1:", x1, "x2:", x2)


def saveAndSortPredictions(trackPred):
    incomplete = 0
    if len(trackPred) < 500:
        incomplete += 1



def validateSzenarios():
    szenarios = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, -1: 0}
    mpd_slice = getDataFromJson(filename="challenge_set.json", path='../')
    for playlist in mpd_slice["playlists"]:
        szenarios[checkSzenario(playlist)] += 1
    print(szenarios)


def checkSzenario(playlist):
    if playlist["num_samples"] == 0:  # Title only (no tracks)
        return 1
    elif playlist["num_samples"] == 1:  # Title and first track
        return 2
    elif playlist["num_samples"] == 5 and len(playlist) == 6:  # Title and first 5 tracks

        return 3
    elif playlist["num_samples"] == 5 and len(playlist) == 5:  # First 5 tracks only
        return 4
    elif playlist["num_samples"] == 10 and len(playlist) == 6:  # Title and first 10 tracks
        return 5
    elif playlist["num_samples"] == 10 and len(playlist) == 5:  # First 10 tracks only
        return 6
    elif playlist["num_samples"] == 25:  # Title and first 25 tracks
        i = 0
        for track in playlist["tracks"]:
            if track["pos"] != i:  # Title and 25 random tracks
                return 8
            i += 1
        return 7  # Title and first 25 tracks
    elif playlist["num_samples"] == 100:  # Title and first 25 tracks
        i = 0
        for track in playlist["tracks"]:
            if track["pos"] != i:  # Title and 100 random tracks
                return 10
            i += 1
        return 9  # Title and first 100 tracks
    return -1


recommendation()

