from helperMethods import readDF_from_CSV, getDataFromJson, normalize_name
import pandas as pd


def ruleDict(filename):
    df = readDF_from_CSV(filename, path='../data_rules/')
    df['rule'] = df[['consequent', 'confidence']].values.tolist()
    df_new = df.groupby('antecedent')['rule'].apply(list).reset_index(name='rules')
    return {row['antecedent']: row['rules'] for index, row in df_new.iterrows()}


def predict(dict, key):
    try:
        consequents = dict[key]
    except:
        return {}

    return {consequent[0] for consequent in consequents}


def recommendation():
    mpd_slice = getDataFromJson(filename="challenge_set.json", path='../')
    print("Load 1000_name2track_uri.csv...", end='')
    pname2track_uri = ruleDict('1000_name2track_uri.csv')
    print(" -> Done!\nLoad 1000_track_uri2track_uri.csv", end='')
    track_uri2track_uri = ruleDict('1000_track_uri2track_uri.csv')
    print(" -> Done")

    emptys = 0
    count = 0
    for playlist in mpd_slice['playlists']:
        print(count)
        szenario = checkSzenario(playlist)
        predictions = set()
        if szenario == 1 or szenario == 2:
            predictions.update(predict(pname2track_uri, normalize_name(playlist['name'])))

            for track in playlist['tacks']:
                predictions.update(predict(track_uri2track_uri, track['track_uri']))

            if len(predictions) == 0:
                emptys += 1

            count += 1

    print(f'{emptys} empty from {count}')


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
            if track["pos"] != i:       # Title and 25 random tracks
                return 8
            i += 1
        return 7                        # Title and first 25 tracks
    elif playlist["num_samples"] == 100:  # Title and first 25 tracks
        i = 0
        for track in playlist["tracks"]:
            if track["pos"] != i:       # Title and 100 random tracks
                return 10
            i += 1
        return 9                        # Title and first 100 tracks
    return -1


recommendation()
