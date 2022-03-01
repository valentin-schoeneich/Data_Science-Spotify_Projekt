import pandas as pd
from helperMethods import readDF_from_CSV, getDataFromJson, normalize_name, appendList, normalize_uri, saveDF2CSV, Path
import statistics
import pandas as pd

submission = list()  # list with all prediction entries


def getRuleDict(filename):
    """
    Method to create a rule dictionary for a given file
    :param filename:    CSV file with rules
    :return         dictionary
    """
    print(f"Load {filename}...", end='')
    df = readDF_from_CSV(filename, path=Path.pathToRules.value)  # create dataframe
    # create entry with a list who includes consequents and confidences
    df['rule'] = df[['consequent', 'confidence']].values.tolist()
    # all rules group by antecedent
    df_new = df.groupby('antecedent')['rule'].apply(list).reset_index(name='rules')
    print(" -> Done!")
    return {row['antecedent']: row['rules'] for index, row in df_new.iterrows()}


def getRuleDicts(item):
    """
    Method to create rule dictionaires for name2item and item2item
    :param item:    artist, album or track
    :return         two dictionaries
    """
    pname2item = getRuleDict(f"1000_name2{item}.csv")
    item2item = getRuleDict(f"1000_{item}2{item}.csv")
    return pname2item, item2item


def getMostPopular(filename):
    """
    Method to find the most popular tracks from a artist or album
    :param filename:    file with albums or aritst and their most successful songs
    :return:            Union with a list and a dictionary
    """
    print(f"Load {filename}...", end='')
    # creates dataframe from csv file
    df = readDF_from_CSV(filename, path=Path.pathToRules.value)
    if len(df.columns) == 1:
        return df['0'].tolist()

    df['0'] = df['0'].apply(lambda x: normalize_uri(x))
    df['1'] = df['1'].apply(lambda x: appendList(x))
    print(" -> Done!")
    return {row['0']: row['1'] for index, row in df.iterrows()}


def myUpdate(consDict, extensionDict):  # Add content from extension dictionary
    for key in extensionDict:
        if key in consDict:  # if both have the key use the one with the larger value
            consDict[key] = max(consDict[key], extensionDict[key])
        else:  # add the new entry to consDict
            consDict[key] = extensionDict[key]


def tryPredict(ruleDict, key, givenTracks):
    """
    Method to find matching predictions for a given input value
    :param ruleDict:    dictionary to find matching predictions
    :param key:         item used for prediction
    :param givenTraks:  tracks who are in playlist
    :return:            dictionary with new predicitons
    """
    prediction = dict()
    if key in ruleDict:
        for rule in ruleDict[key]:
            if not rule[0] in givenTracks:
                prediction.update({rule[0]: rule[1]})
    return prediction


def predForTrack(tracks, dictItem, itemName, givenTracks):
    """
    Method to predict new items for given tracks
    :param tracks:        track with attributes
    :param dictItem:      rule dictionary of the item
    :param itemName:      item uri
    :param givenTraks:    tracks uris
    :return:            dirctionary with new predicitons
    """
    predictions = dict()
    for track in tracks:
        # try to predict for every track new items
        myUpdate(predictions, tryPredict(dictItem, track[itemName], givenTracks))
    return predictions


def predForItem(items, dictItem, givenTracks):
    """
    Method to predict new items
    :param items:         items for prediction
    :param dictItem:      rule dictionary of the item
    :param givenTraks:    tracks uris
    :return:              dirctionary with new item predicitons
    """
    predictions = dict()
    for item in items:
        # try to predict for every item in items and add the result to predictions
        myUpdate(predictions, tryPredict(dictItem, item, givenTracks))
    return predictions


def addPopular(trackPred, pred, popDict, maxTracks, givenTracks):
    """
    Method to add most popular songs from album or artist predictions to the track predictions:
    :param trackPred:
    :param pred:         album or artist predictions
    :param popDict:      dictionary with album or artist to most popular songs
    :param maxTracks:    maximum number of tracks that are still required
    :param givenTracks:  tracks from playlist
    """
    for item in pred:
        i = 0
        for track in popDict[item]:
            if i > maxTracks or track in givenTracks:
                break
            i += 1
            myUpdate(trackPred, {track: pred[item]})


def recommendation():
    """
    Main method to predict 500 tracks for all playlists from challenge_set.json.
    :return:    No return value, but calling the saveAndSortPrediction method,
                which creates a CSV submission file with predictions for all playlists.
    """
    # calling the getDataFromJson helper method, which creates a dataframe from the challenge_set.json file
    mpd_slice = getDataFromJson(filename="challenge_set.json", path=Path.pathToChallenge.value)
    # create variables for the most popular songs from all artist and albums
    album2mostPopular = getMostPopular("album_uri2track_uris_sorted.csv")
    print(album2mostPopular)
    artist2mostPopular = getMostPopular("artist_uri2track_uris_sorted.csv")
    # variable with most popular tracks
    mostPopularTracks = getMostPopular("mostPopularTracks.csv")
    # dictonaries for different attribute types
    pname2track, track2track = getRuleDicts('track_uri')
    pname2artist, artist2artist = getRuleDicts('artist_uri')
    pname2album, album2album = getRuleDicts('album_uri')
    # execution control variables
    incomplete = 0
    count = 0
    x1 = 0
    x2 = 0
    lenPredictions1 = list()
    lenPredictions2 = list()
    for playlist in mpd_slice['playlists']:  # iterates through all challenge playlists
        print(count)
        # Method to detect the scenario. Returns the scenario number
        szenario = checkSzenario(playlist)
        # Creates dictionaries for the predictions
        trackPred, albumPred, artistPred = dict(), dict(), dict()
        tracks = playlist['tracks']
        # Creates a set with all track from the playlist
        givenTracks = set()
        for track in tracks:
            givenTracks.add(track['track_uri'])
        if szenario != 4 and szenario != 6:  # playlist-name given
            # normalize to find more overlap in names
            pname = normalize_name(playlist['name'])
            # make predictions from playlist-name for albums, artists and tracks
            myUpdate(albumPred, tryPredict(pname2album, pname, givenTracks))
            myUpdate(artistPred, tryPredict(pname2artist, pname, givenTracks))
            myUpdate(trackPred, tryPredict(pname2track, pname, givenTracks))
        # make predictions from tracks
        myUpdate(albumPred, predForTrack(tracks, album2album, 'album_uri', givenTracks))
        myUpdate(artistPred, predForTrack(tracks, artist2artist, 'artist_uri', givenTracks))
        myUpdate(trackPred, predForTrack(tracks, track2track, 'track_uri', givenTracks))
        lenPredictions1.append(len(trackPred))
        # if no predictions are found, the top 500 tracks from all playlists are used as predictions
        if len(trackPred) == 0 and len(albumPred) == 0 and len(artistPred) == 0:
            print("Found", len(trackPred), "predictions for playlist:", playlist)
            myUpdate(trackPred, {track: 0 for track in mostPopularTracks[0:500]})
            incomplete += 1
        for i in range(0, 4):  # fill predictions
            kmax = (500 - len(trackPred)) // max(len(albumPred) + len(artistPred), 1)
            # fill track prediction wit most popular tracks from artist and album predictions
            addPopular(trackPred, albumPred, album2mostPopular, kmax, givenTracks)
            addPopular(trackPred, artistPred, artist2mostPopular, kmax, givenTracks)
            if i == 0:
                lenPredictions2.append(len(trackPred))
            if len(trackPred) >= 500:  # enough predictions
                x1 += 1
                break
            # new item prediction
            albumPred.update(predForItem(albumPred, album2album, givenTracks))
            artistPred.update(predForItem(artistPred, artist2artist, givenTracks))
            trackPred.update(predForItem(trackPred, track2track, givenTracks))

            if len(trackPred) >= 500:  # enough predictions
                x2 += 1
                break
        # # if not enough predictions are found
        if len(trackPred) < 500:
            incomplete += 1
        i = 0
        # fill predictions with top tracks
        while len(trackPred) < 500:
            myUpdate(trackPred, {mostPopularTracks[i]: 0})
            i += 1

        count += 1

        # call method to save predictions
        #saveAndSortPredictions(trackPred, count == 10000, playlist['pid'])

    print(f'{incomplete} incomplete from {count}')
    print("Avg 1: ", statistics.mean(lenPredictions1))
    print("Avg median 1: ", statistics.median(lenPredictions1))
    print("Avg 2: ", statistics.mean(lenPredictions2))
    print("Avg median 2: ", statistics.median(lenPredictions2))
    print("x1:", x1, "x2:", x2)


def saveAndSortPredictions(trackPred, finish, pid):
    """
        :param trackPred:   Dictionary with 500 predictions
        :param finish:      Boolean which is true if all predictions have been made
        :param pid:         Playlist-Id
        :return:            No return value, but creates CSV-File if called last time
        """
    tracks = list(trackPred.keys())
    tracks.sort(key=lambda x: trackPred[x], reverse=True)  # sort the tracks
    row = [pid]  # new list with pid as the first entry
    row.extend(tracks[0:500])  # append the sorted predictions
    submission.append(row)  # append playlist prediction to the global submission list
    if finish:  # if all playlist predicted create submission file
        df = pd.DataFrame(submission)
        saveDF2CSV(df, 'submission.csv', path='../', header=False)


def validateSzenarios():  # just to check the correct number of scenarios
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
