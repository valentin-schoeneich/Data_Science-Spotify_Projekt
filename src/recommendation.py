import pandas as pd
from helperMethods import readDF_from_CSV, getDataFromJson, normalize_name, appendList, normalize_uri, saveDF2CSV
import statistics
import pandas as pd

submission = list()


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


def myUpdate(consDict, extensionDict):
    for key in extensionDict:
        if key in consDict:
            consDict[key] = max(consDict[key], extensionDict[key])
        else:
            consDict[key] = extensionDict[key]


def tryPredict(ruleDict, key, givenTracks):
    prediction = dict()
    if key in ruleDict:
        for rule in ruleDict[key]:
            if not rule[0] in givenTracks:
                prediction.update({rule[0]: rule[1]})
    return prediction


def predForTrack(tracks, dictItem, itemName, givenTracks):
    predictions = dict()
    for track in tracks:
        myUpdate(predictions, tryPredict(dictItem, track[itemName], givenTracks))
    return predictions


def predForItem(items, dictItem, givenTracks):
    predictions = dict()
    for item in items:
        myUpdate(predictions, tryPredict(dictItem, item, givenTracks))
    return predictions


def addPopular(trackPred, pred, popDict, maxTracks, givenTracks):
    for item in pred:
        i = 0
        for track in popDict[item]:
            if i > maxTracks or track in givenTracks:
                break
            i += 1
            myUpdate(trackPred, {track: pred[item]})


def recommendation():
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
        trackPred, albumPred, artistPred = dict(), dict(), dict()
        tracks = playlist['tracks']
        givenTracks = set()
        for track in tracks:
            givenTracks.add(track['track_uri'])

        if szenario != 4 and szenario != 6:  # playlist-name given
            pname = normalize_name(playlist['name'])
            # make predictions from playlist-name
            myUpdate(albumPred, tryPredict(pname2album, pname, givenTracks))
            myUpdate(artistPred, tryPredict(pname2artist, pname, givenTracks))
            myUpdate(trackPred, tryPredict(pname2track, pname, givenTracks))
        # make predictions from tracks
        myUpdate(albumPred, predForTrack(tracks, album2album, 'album_uri', givenTracks))
        myUpdate(artistPred, predForTrack(tracks, artist2artist, 'artist_uri', givenTracks))
        myUpdate(trackPred, predForTrack(tracks, track2track, 'track_uri', givenTracks))

        if len(trackPred) == 0 and len(albumPred) == 0 and len(artistPred) == 0:
            print("Found", len(trackPred), "predictions for playlist:", playlist)
            myUpdate(trackPred, {track: 0 for track in mostPopularTracks[0:500]})
            incomplete += 1

        for i in range(0, 4):
            kmax = (500 - len(trackPred)) // max(len(albumPred) + len(artistPred), 1)
            addPopular(trackPred, albumPred, album2mostPopular, kmax, givenTracks)
            addPopular(trackPred, artistPred, artist2mostPopular, kmax, givenTracks)

            if len(trackPred) >= 500:
                x1 += 1
                break

            albumPred.update(predForItem(albumPred, album2album, givenTracks))
            artistPred.update(predForItem(artistPred, artist2artist, givenTracks))
            trackPred.update(predForItem(trackPred, track2track, givenTracks))

            if len(trackPred) >= 500:
                x2 += 1
                break

        if len(trackPred) < 500:
            incomplete += 1
        i = 0
        while len(trackPred) < 500:
            myUpdate(trackPred, {mostPopularTracks[i]: 0})
            i += 1

        count += 1
        lenPredictions.append(len(trackPred))
        saveAndSortPredictions(trackPred, count == 10000, playlist['pid'])

    print(f'{incomplete} incomplete from {count}')
    print("Avg: ", statistics.mean(lenPredictions))
    print("max: ", max(lenPredictions))
    print("x1:", x1, "x2:", x2)


def saveAndSortPredictions(trackPred, finish, pid):
    tracks = list(trackPred.keys())
    tracks.sort(key=lambda x: trackPred[x], reverse=True)
    row = [pid]
    row.extend(tracks[0:500])
    submission.append(row)
    if finish:
        df = pd.DataFrame(submission)
        saveDF2CSV(df, 'submission.csv', path='../', header=False)


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

