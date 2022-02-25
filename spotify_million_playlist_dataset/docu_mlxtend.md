## Dokumentation: Versuche mit mlxtend
Um eine Assoziationsanalyse über unsere Spotofy Daten durchzuführen liegt es
Nahe eine Library zu verwenden, welche die benötigte Funktionalität bereits 
bereitstellt. Wir sind dabei auf Library mlxtend gestoßen, die z.B. Apriori [1] und Funktionen
zur Generierung von Assoziationsregeln [2] implementiert.

### Data-Input
Die benötigten Daten fragen wir aus unserer Datenbank ab. Bevor wir aber die Daten an die
entsprechende mlxtend Funktion übergeben können, müssen diese in das passende Format gebracht
werden. Dazu befindet sich in src/db die Funktion `getC1ItemSets` welche die Parameter item und
maxPid erwartet. Wir können also als item z.B. track_uri und als maxPid 3 übergeben. Dann
wird eine Anfrage an die Datenbank gesendet, um alle track_uri aus den ersten 3 Playlists zu
erhalten. 

    item = "track_uri"
    maxPid = 3
    _dbReq(f"SELECT string_agg(x.{item}::character varying, ',') "
                  f"FROM ({_getQueryUniqueItemsOfPlaylists(item, maxPid)}) AS x "
                  f"GROUP BY pid")

    = SELECT string_agg(x.track_uri::character varying, ',')
      FROM (SELECT track_uri, pid FROM pcont WHERE pid < 3 GROUP BY track_uri, pid) AS x
      GROUP BY pid

    = [('spotify:track:2nVHqZbOGkKWzlcy1aMbE7,spotify:track:1NXTEkIeRL59NK61QuhYUl,...),
       ('spotify:track:4E5P1XyAFtrjpiIxkydly4,spotify:track:1Y4ZdPOOgCUhBcKZOrUFiS,...),
       ('spotify:track:2SYa5Lx1uoCvyDIW4oee9b,spotify:track:1enx9LPZrXxaVVBxas5rRm,...)]

Zunächst führen wir eine Unterabfrage durch, die von der Funktion `_getQueryUniqueItemsOfPlaylists`
generiert wird. Wir erhalten eine Liste der Tracks aus den ersten 3 Playlist jeweils als
Tupel, wie unter dargestellt. Anschließend fügen wir mit der SQL-Funktion `string_agg` die
Tracks in der Spalte `track_uri` gruppiert nach der pid zusammen. Der Output wird eine Liste 
aus 3 Playlists mit den darin enthaltenen Tracks sein.
Diese SQL Select-Abfrage wird nun von der Funktion `_dbReq` verarbeitet, wo die Verbindung zur 
Datenbank hergestellt wird und die Anfrage versendet wird.

    _getQueryUniqueItemsOfPlaylists("track_uri", 3)
    = SELECT track_uri, pid FROM pcont WHERE pid < 3 GROUP BY track_uri, pid
    = [('spotify:track:2nVHqZbOGkKWzlcy1aMbE7', 1), ('spotify:track:4E5P1XyAFtrjpiIxkydly4', 0),...]

Jetzt bauen wir den Output noch so um, dass wir eine Liste die wiederum Listen als Items
enthält erhalten. Jede Liste repräsentiert eine Playlist und enthält Tracks als Items.

    dataset = [['Track0', 'Track1', 'Track2', 'Track3', 'Track4', 'Track5'],
               ['Track2', 'Track6', 'Track7', 'Track4', 'Track8', 'Track5'],
               ['Track6', 'Track1', 'Track2', 'Track7'],
               ['Track6', 'Track4', 'Track0', 'Track8']]

Jetzt sind die Daten in dem für mlxtend erforderlichen Format bereit für die weitere Analyse.
Anmerkung: Analog zu `track_uri`, kann man die Abfragen auch für `album_uri` und `artist_uri`
generieren um dann im weiteren Verlauf Regeln für Ablbums und Artists zu erzeugen.

### Die mlxtend Library
Zunächst überführen wir die Werte weiter in eine binäre Darstellung. Ein Track ist entweder in
der Playlist enthalten, dann ist der Wert in der Spalte True, ansonsten false. Dies erreichen
wir durch den TransactionEncoder von mlxtend. Dann erzeugen wir aus der Liste wieder einen
Dataframe.

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

Der beispielhafte Dataframe sieht nun folgendermaßen aus:

        Track0  Track1  Track2  Track3  Track4  Track5  Track6  Track7  Track8
    0    True    True    True    True    True    True   False   False   False
    1   False   False    True   False    True    True    True    True    True
    2   False    True    True   False   False   False    True    True   False
    3    True   False   False   False    True   False    True   False    True

Die Funktion `fpgrowth` erzeugt nun alle Itemsets und berechnet den Support. Für unsere
Beispiel-Daten sagen wir die Items bzw Itemsets sollen in 2 von 4 Playlists vorkommen, der
Support ist also 50%. Den gleichen Output würde auch die mlxtend-Funktion
`apriori(df, min_support=0.5)` erzeugen. Jedoch ist `fpgrowth` für große Datenmenge besser 
geeignet, da es im Gegensatz zu apriori keine Kandidaten erzeugt. Stattdessen verwendet es die
frequent pattern tree Datenstruktur.

    frequentItemSets = fpgrowth(df, min_support=0.5)

         support   itemsets
    0      0.50        (0)
    1      0.50        (1)
    2      0.75        (2)
    ...
    19     0.50  (2, 4, 5)
    20     0.50  (2, 6, 7)
    21     0.50  (8, 4, 6)

Mit den `frequentItemSets` können wir schließlich die Regeln generieren. Wir rufen dazu die 
mlxtend-Funktion `association_rules` auf und übergeben `frequentItemSets`. Zusätzlich bestimmen
wir das wir über die Konfidenz ermittelt werden sll, ob eine Regel interessant ist.
`min_threshold` ist hierbei der Wert für die Konfidenz.

    rules = association_rules(frequentItemSets, metric="confidence", min_threshold=0.7)

        antecedents consequents  antecedent support  
    0          (5)         (2)                 0.5
    1          (5)         (4)                 0.5 
    2       (2, 4)         (5)                 0.5 
    ...
    17      (2, 7)         (6)                 0.5
    18      (6, 7)         (2)                 0.5
    19         (7)      (2, 6)                 0.5

### Probleme mit mlxtend
Zunächst wollten wir den Algorithmus für wenige Playlists testen. Dafür haben wir 100
Playlists geladen und mlxtend hat schon über 250.000 Itemsets generiert. Deshalb haben wir
in der mlxtend-Funktion `apriori` den Parameter `max_len` auf 2 gesetzt um nur noch Itemsets
der Länge 2 zu generieren. \
Für 10.000 Playlists funktionierte das auch nur noch, wenn wir
zusätzlich den Parameter `low_memory` auf True setzten. Dies hat laut der mlxtend-Dokumentation
zur Folge, dass der Algorithmus 3-6x länger braucht aber Speicher-schonend arbeitet. \
Bei 100.000 Plylists war dann selbst mit `low_memory=True` und `max_len=2` der Arbeitsspeicher
derat ausgelastet, dass das Program mit Swapping beginnen musste.

### References
[1] http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/ \
[2] http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

