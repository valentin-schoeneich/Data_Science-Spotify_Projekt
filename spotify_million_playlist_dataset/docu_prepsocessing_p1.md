## Preprocessing: Teil 1
Zu Beginn unserer Arbeit haben wir uns überlegt den vollständigen 33GB großen Datensatz
vom JSON-Format in das CSV-Format zu überführen. Dies hat einerseits den Vorteil, dass der
Speicherbedarf der CSV-Dateien wesentlich geringer ist. Im Json-Format ist eine große Menge an
Redundanz durch Bezeichner wie "track_uri" gegeben die im CSV-Dateien nur einm einziges Mal
im Header gespeichert sind. Der zweite Vorteil ist, dass man CSV mit einem einzigen Befehl in
PostgreSQL Datenbanken laden kann.

### DB-Schema
Bevor wir unsere Daten transformiert haben, haben wir schrittweise ein geeignetes
Datenbank-Schema entwickelt. Zunächst hatten wir ein Star-Schema, dass sich in der
Nachbesprechung als ineffizient bei Abfragen herausstellte. Das Problem war, dass die
Verknüpfungstabelle "soulOfStar" mit all ihren Attributen zu breit war und bei einer
Länge von 66mio Zeilen zu groß war um diese schnell zu durchlaufen.
<img src="./images/DB-Schema-v1.png" alt="DB_Schema-Version-1.png">
Aufgrund der schlechten Performance haben wir das Schema dann überarbeitet und primär die 
Zuordnungstabelle schmaler gebaut. Diese ist im Schema mit der n:m-Beziehung `contains`
dargestellt und speichert eine Playlist-ID (pid), die jeweilligen `track_uri` und `pos`,
also Position des Tracks in der Playlist.
<img src="./images/DB-Schema-v2.png" alt="DB_Schema-Version-2.png">

### Generierung CSV-Dateien
Wir müssen also nun die JSON-Daten in 5 CSV-Dateien mit Python transformieren:

* tracks
* artists
* albums
* playlists
* playlist contains tracks (im folgenden mit pConT abgekürzt)

Dafür haben wir eine Funktion `csvForDb(maxFiles)` geschrieben, die `maxFiles` viele der 
1000 JSON-Dateien konvertiert, konkateniert und in einer der fünf CSV-Dateien speichert.
Wir lesen alle Dateien aus dem Quellordner der JSON-Dateien ein:

    filenames = os.listdir(pathToData)
    for filename in sorted(filenames, key=natural_keys):
        if not filename.startswith('mpd.slice.') and filename.endswith(".json") or i > maxFiles:
            break

Dann rufen wir eine zweite Funktion `createDfsForDb(filename)` auf, welche die Daten in einen
pandas Dataframe lädt. Die Funktion gibt eine Liste der fünf Dataframes zurück. Dann können
die entsprechenden Dataframes in CSV-Dateien gespeichert werden:

    for df, name in zip(createDfsForDb(filename), ["playlists", "artists", "tracks", "albums", "pConT"]):
                saveDF2CSV(df, f'{name}_{maxFiles}.csv', mode='a', header=(i == 1))

Der wichtigste Teil unserer Funktion `createDfsForDb(filename)` ist das flatten der JSON-Daten
mit der pandas Funktion `json_normalize`. Die JSON-Dateien öffnen wir erst per File-Deskriptor
und verarbeiten diese dann mit `json_normalize`. Als `record_path` geben wir der Funktionan,
was wir aus der JSON-Datei flatten möchten und erhalten den Dataframe, den wir im folgenden
eben in die fünf Dataframes umbauen und zurückgeben.

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

Da wir nun jede Playlist eingelesen haben, existieren noch unzählige Duplikate in unseren 
CSV-Dateien, die wir bequem mit der Funktion `df.drop_duplicates()` entfernen konnten.

### CSV-Dateien in die DB laden
Die CSV-Dateien haben wir dann sehr einfach mit z.B. folgenden Befehl in die Datenbank laden
können:

    \copy tracks FROM '/.../tracks_1000.csv' DELIMITER ',' CSV HEADER

Dabei war darauf zu achten die Tabellen in der richtigen Reihenfolge zu befüllen, da manche 
von einer anderen abhängig waren durch Fremdschlüssel-Referenzen.

### Daten aus der DB abfragen
Für unseren ersten Ansatz, den Apriori aus dem Git-Repo zu verwenden, haben wir uns eine 
Funktion `csvForDb(maxFiles)` gebaut. Diese fragt die entsprechenden Daten aus der DB ab und 
bringt diese ins erste benötigte Input-Format für den Algorithmus. Dazu fragen wir mit folgendem
Select `track_uri` und `pid` ab und erhalten eine Liste aus Tupeln:
    
    tracks = _dbReq(f'SELECT track_uri, pid FROM pcont WHERE pid<{maxPid}')

Die Funktion `_dbReq` wird bei `Ansatz III - Library von mlxtend verwenden` genauer erläutert.
        
        playlists = []
        unique_tracks = set()
        playlistCounter = 0
        record = set()
        
        for track in tracks:
            if playlistCounter != track[1]:  # switch from playlist i to playlist i+1
                playlistCounter += 1
                playlists.append(record)
                record = set()
            track_uri = track[0]
            record.add(track_uri)
            unique_tracks.add(frozenset([track_uri]))
            playlists.append(record)
        return unique_tracks, playlists

Dann interieren wir über alle angefragten tracks, müssen aber wieder tracks in die korrekte
Playlist einordnen. Deshalb haben wir auch `pid` mit ausgeben lassen und können so wieder
erfassen welcher track zu welcher playlist gehört. Dann erstellen wir für jede Playlist ein set
und speichern diese in eine Liste. Zusätzlich generieren wir eine Liste mit allen Tracks, die unique
ist.