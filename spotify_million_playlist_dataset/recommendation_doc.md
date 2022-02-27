## Vorhersagen
Die Datei `recommendation.py` ist dafür zuständig, aus den vorher mit dem Apriori
angelegten Regeln eine CSV-Datei zu erstellen, die zu jeder Playlist aus der
`chalenge_set.json` Datei, 500 Lieder vorschlägt.

### `recommendation`
Die Recommendation-Methode ist unsere Hauptmethode zum Vorhersagen unserer Songs. Als erstes lädt sie
die erstellten CSV-Dateien mit den ausgewählten Regeln in CSV-Dateien:

    mpd_slice = getDataFromJson(filename="challenge_set.json", path='../')

    album2mostPopular = getMostPopular("album_uri2track_uris_sorted.csv")
    artist2mostPopular = getMostPopular("artist_uri2track_uris_sorted.csv")
    mostPopularTracks = getMostPopular("mostPopularTracks.csv")

    pname2track, track2track = getRuleDicts('track_uri')
    pname2artist, artist2artist = getRuleDicts('artist_uri')
    pname2album, album2album = getRuleDicts('album_uri')

Dann wird in einer Schleife alle Playlists durchgelaufen, um für jede eine 
Vorhersage durchzuführen:

    for playlist in mpd_slice['playlists']:

Als Erstes wird für jede Playlist kontrolliert, in welchem Szenario diese sich
befindet. Dies geschieht durch die `checkSzenario` Methode:

    szenario = checkSzenario(playlist)

Die Szenarien sind in den Vorgaben der Challenge vorgegeben und wir nutzen sie um zuerkennen
ob es einen Playlist-Namen gibt, den wir auch zur Vorhersage nutzen.

Als Erstes werden Vorhersagen anhand des Namens erstellt, außer es ist keiner
vorhanden (Szenario 4 und 6):

    myUpdate(albumPred, tryPredict(pname2album, pname, givenTracks))
    myUpdate(artistPred, tryPredict(pname2artist, pname, givenTracks))
    myUpdate(trackPred, tryPredict(pname2track, pname, givenTracks))

Dann werden Vorhersagen anhand der Songs getätigt. Hierbei werden die direkten
Regeln von Songs und die Regeln zu Künstler und Artist genutzt:

    myUpdate(albumPred, predForTrack(tracks, album2album, 'album_uri', givenTracks))
    myUpdate(artistPred, predForTrack(tracks, artist2artist, 'artist_uri', givenTracks))
    myUpdate(trackPred, predForTrack(tracks, track2track, 'track_uri', givenTracks))

Wenn nach den Vorhersagen zu dem Namen und Songs keine Ergebnisse zustande
gekommen sind, nutzen wir die am meisten vorkommenden Song unseres gesamten Datensatzes:

    if len(trackPred) == 0 and len(albumPred) == 0 and len(artistPred) == 0:
        print("Found", len(trackPred), "predictions for playlist:", playlist)
        myUpdate(trackPred, {track: 0 for track in mostPopularTracks[0:500]})

Danach wird eine Schleife viermal durchgelaufen, die zum Auffüllen unserer Vorhersagen dient. 
Sobald wir die Anforderung von 500 Song erreicht haben verlassen wir die Schleife sofort:

    for i in range(0, 4):
        kmax = (500 - len(trackPred)) // max(len(albumPred) + len(artistPred), 1)
        addPopular(trackPred, albumPred, album2mostPopular, kmax, givenTracks)
        addPopular(trackPred, artistPred, artist2mostPopular, kmax, givenTracks)
        if len(trackPred) >= 500: # enough predictions
            x1 += 1
            break
        albumPred.update(predForItem(albumPred, album2album, givenTracks))
        artistPred.update(predForItem(artistPred, artist2artist, givenTracks))
        trackPred.update(predForItem(trackPred, track2track, givenTracks))
        if len(trackPred) >= 500: # enough predictions
            x2 += 1
            break

Wenn auch diese Schleife es nicht geschafft hat genug Vorhersagen zu erstellen, erweitern
wir unsere Vorhersagen mit den Top Songs bis wir die 500 Songs erreicht haben.
    
    while len(trackPred) < 500:
        myUpdate(trackPred, {mostPopularTracks[i]: 0})
        i += 1

Am Ende jeder Playlist-Vorhersage wird die Methode `saveAndSortPrediction` aufgerufen,
welche unsere Vorhersagen sortiert, an globale Liste anfügt und beim letzten Aufruf
unsere gesamten Vorhersagen in einer `submission.csv` Datei speichert:

    saveAndSortPredictions(trackPred, count == 10000, playlist['pid'])

### `getRuleDicts`
Methode zum Erstellen von Dictionaries aus unseren vorher angelegten Regeln:

    def getRuleDicts(item): 
        pname2item = getRuleDict(f"1000_name2{item}.csv")
        item2item = getRuleDict(f"1000_{item}2{item}.csv")
        return pname2item, item2item

### `getMostPopular`
Diese Methode ist dafür zuständig die erfolgreichsten Tracks eines Albums oder Artist zu ermitteln:

    print(f"Load {filename}...", end='')
    df = readDF_from_CSV(filename, path='../data_rules/')
    if len(df.columns) == 1:
        return df['0'].tolist()

    df['0'] = df['0'].apply(lambda x: normalize_uri(x))
    df['1'] = df['1'].apply(lambda x: appendList(x))
    print(" -> Done!")
    return {row['0']: row['1'] for index, row in df.iterrows()}

### `myUpdate`
Methode zum Aktualisieren eines Dictionaries, durch ein anderes. Hierbei werden neue Einträge
eingefügt und bereits vorhandene Einträge ersetzt, wenn Eintrag in einem anderem Dictionary stärker
ist:

    def myUpdate(consDict, extensionDict):
        for key in extensionDict:
            if key in consDict:
                consDict[key] = max(consDict[key], extensionDict[key])
            else:
                consDict[key] = extensionDict[key]

### `tryPredict`
Diese Methode nutzen wir unter anderem zur Vorhersage von Tracks, Artist und Alben anhand des Playlist-Namens.
Sie gibt ein Dictionary zurück welches die passenden Vorhersagen beinhaltet:

    def tryPredict(ruleDict, key, givenTracks):
        prediction = dict()
        if key in ruleDict:
            for rule in ruleDict[key]:
                if not rule[0] in givenTracks:
                    prediction.update({rule[0]: rule[1]})
        return prediction

### `predForTrack`
Methode zum Erstellen eines Dictionaries mit Song-Vorhersagen zu dem jeweiligen Item:

    predictions = dict()
    for track in tracks:
        myUpdate(predictions, tryPredict(dictItem, track[itemName], givenTracks))
    return predictions

### `predForItem`
Methode zum Erstellen von Item-Vorhersagen:

    predictions = dict()
    for item in items:
        myUpdate(predictions, tryPredict(dictItem, item, givenTracks))
    return predictions

### `addPopular`
Diese Methode nutzen wir um unsere Vorhersagen mit den erfolgreichsten Tracks der vorher durchgeführten
Album und Artist Vorhesagen zu füllen.

    for item in pred:
        i = 0
        for track in popDict[item]:
            if i > maxTracks or track in givenTracks:
                break
            i += 1
            myUpdate(trackPred, {track: pred[item]})

### `saveAndSortPredictions`
Methode zum Speichern der Vorhersagen mit Playlist-ID in einer globalen 
Submission-Liste. Beim letzten Durchlauf wird eine CSV Datei mit allen Ergebnissen
erstellt.

    tracks = list(trackPred.keys())
    tracks.sort(key=lambda x: trackPred[x], reverse=True) 
    row = [pid] 
    row.extend(tracks[0:500]) 
    submission.append(row) 
    if finish: 
        df = pd.DataFrame(submission)
        saveDF2CSV(df, 'submission.csv', path='../', header=False)

### `checkSzenario`
Diese Methode nutzen wir zum Überprüfen in welchem Szenario sich die Playlist befindet.

