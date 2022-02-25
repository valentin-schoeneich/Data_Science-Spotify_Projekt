# <center> Data Science WS 2021/2022 <br/> Dokumentation - Lösung der AI crowd Spotify Million Playlist Dataset Challenge mittels Assoziationsanalyse

## Begrifflichkeiten
Um das grundsätzliche Verständnis zum Code und dieser Dokumentation zu fördern, erläutern wir im folgenden häufig verwendete Bergriffe bzw. Variablennamen:
* `items` = {`track_name`, `track_uri`, `artist_uri`,`album_uri`, `name` (Playlist-name), `pid`}
* `itemSet`: Eine Menge unterschiedlicher items gleicher Art. Meistens in Form eines `frozensets` um sie einem `set` anzufügen zu können.
* `c1ItemSet`: ItemSets der Länge 1, welche auch unter `minSup` liegen
* `l1ItemSet`: ItemSets der Länge 1, welche über `minSup` liegen
* `l2ItemSet`: ItemSets der Länge 2, welche über `minSup` liegen
* `antecedents`: Eine Menge von Vorgängern in Form von `itemSets`, von denen ausgehend eine Vorhersage getroffen werden kann.
* `consequents`: Die Vorhersage (auch Nachfolger) in Form einer Menge von `itemSets`, welche durch eine Regel erzeugt werden kann.
* `confidence`: Die Konfidenz steht für die Wahrscheinlichkeit, dass die Nachfolger eintreten, gegeben der Wahrscheinlichkeit, dass die Vorgänger bereits eingetroffen sind.
* `minSup`: Abkürzung von minimum Support. Bezeichnet die totale Anzahl von Transaktionen bzw. Playlists in denen ein `itemSet` vorkommen soll.  
* `rule` = `[{A, ...}, {B, ...}, Konfidenz]` bzw. `[antecedent, consequent, confidence]`  
  In den meisten Fällen bestehen beide Mengen aus nur einem Element. Eine Regel könnte also lauten:  
  `[{A}, {B}, 0.5]`.  
  In diesem Fall kann die Regel wie folgt gelesen werden:  
  "Wenn der Track A in der Playlist enthalten ist, wird zu 50% auch Track B enthalten sein."   
* `candidates`: Bezeichnet eine Menge von `itemSets`, welche noch nicht auf `minSup` geprüft ist.
* **frequent**: `itemSets` werden frequent genannt, wenn sie über `minSup` liegen.
* `maxPid`: Definiert die größtmögliche ***pid***, mit welcher das Programm rechnen soll. Dadurch wird die größe des Datensatzes bestimmt.  
  Beispiel: Liegt `maxPid` bei 1000, wird mit 1000 Playlists gerarbeitet.

## Ziel
Um die Challenge zu bewältigen, gilt es zwei Anforderungen an die Regeln zu stellen.
1. Wir benötigen mehrere Regeln pro Track. Je nach Szenario und Recommendation-Verfahren schwankt die Anzahl der Regeln, 
   die wir für einen einzelnen Track benötigen. Es gilt, je weniger Tracks wir gegeben haben, desto mehr Regeln 
   benötigen wir für einen einzelnen Track bzw. Playlist-Namen.
2. Wir benötigen für möglichst viele Tracks eine Regel. Je nach Szenario haben wir nur einen Track gegeben, 
   was bedeutet, dass wir für genau diesen Track mehrere Regeln gespeichert haben müssen.
   
Zudem gehen wir davon aus, dass wir nicht wissen für welche Tracks wir eine Vorhersage treffen sollen. Das heißt, 
wir speichern bestenfalls für jeden der ca. 2 Mio. Tracks aus unserem Datensatz eine Regel.

Um die Anzahl derer Tracks zu maximieren, für die wir eine Regel zur Verfügung haben, dient uns als wichtigster 
Parameter `minSup`. Um so kleiner `minSup` gewählt wird, für umso mehr Tracks erhält man eine Regel.  
Damit wir wissen, wie groß wir `minSup` wählen können, um dennoch genug Regeln zu erstellen, haben wir uns eine Methode 
`printSupInfo(item)` geschrieben, welche auflistet, wie oft welcher Support vorkommt.

```
printSupInfo('track_uri')
```

<img src="./images/sup_track_uri.png" alt="sup_track_uri.png">

Wie man in der vierten Spalte erkennen kann, haben 47% aller Tracks nur einen Support von 1. 
Für all diese Tracks lassen sich daher keine aussagekräftigen Regeln bilden. Wir haben uns gefragt, ob das eventuell an 
der Eindeutigkeit der ***track_uri*** liegt und eigentlich gleiche Tracks wegen unterschiedlicher Versionen
oder doppelten Uploads nicht als mehrfach gezählt werden können.
Wir haben die Methode deshalb auch für den ***track_name*** durchlaufen lassen, welchen wir zusammen setzten aus
`track_name + " by " + artist_name`.

```
printSupInfo('track_name')
```

<img src="./images/sup_track_name.png" alt="sup_track_name.png">

Das Ergebnis ist leider ziemlich das gleiche.  
Wir haben die Methode auch für die Künstler, Alben und Playlist-Namen aufgerufen:

```
printSupInfo('artist_uri')
```

<img src="./images/sup_artist_uri.png" alt="sup_artist_uri.png">

```
printSupInfo('album_name')
```

<img src="./images/sup_album_uri.png" alt="sup_album_uri.png">


Das Ergebnis für die Alben und Künstler ist ähnlich, auch für diese lassen sich nicht viel mehr Tracks für Regeln finden.
Wir haben die Methode auch für ***album_name*** und ***artist_name*** aufgerufen, aber auch für diese Attribute war die Ausgabe
ähnlich.
Daraus lässt sich ableiten, dass sich für Rund die Hälfte der Tracks aus unserem Lerndatensatz keine
aussagekräftigen Regeln ableiten lassen.

Für die Playlist-Namen ist der prozentuale Anteil derer, die einen Support von 1 haben jedoch deutlich geringer: 

```
printSupInfo('name')
```

<img src="./images/sup_playlist_name.png" alt="sup_playlist_name.png">

Das lässt vermuten, dass die Assoziationsanalyse für diesen Bestandteil am besten funktionieren wird.




## Preprocessing
## Ansatz I - Vorhandenes Repository umbauen
In unserem ersten Ansatz haben wir versucht ein bestehendes Repository (s. https://github.com/chonyy/apriori_python) 
für unseren Anwendungsfall zu nutzen und dementsprechend umzubauen. 
Diesen Ansatz haben wir in der Datei **apriori_first.py** festgehalten. Leider mussten wir diesen Ansatz verwerfen,
da er für größere Eingaben (ab 2000 Playlists) nicht terminierte.

Das Dokument besteht im Wesentlichen aus drei Methoden:
* `getAboveMinSup(candidateTrackSets, playlistSets, minSup, globalTrackSetWithSup)`:  
  Iteriert für jeden Kandidaten aus `candidateTrackSets` über alle Playlists aus `playlistSets` und zählt dabei, in wie vielen Playlists jeder Kandidat vorkommt.  
  Der Aufwand kann daher wie folgt berechnet werden: `len(playlistSets)` * `len(candidateTrackSets)`.  
* `getUnion(trackSets, length)`: Berechnet aus frequent `trackSets` der Länge k, Kandidaten der Länge k+1, indem sie für jedes `trackSet` über alle `trackSets` iteriert und mit jedem eine Vereinigung bildet.  
  Der Aufwand liegt daher bei `len(trackSets)` * `len(trackSets)`
  
* `aprioriFromDB(maxPlaylists, minSup, minConf=0.5, kMax=2)`: Ruft die zuvor genannten Methoden so oft auf, bis die `itemSets` so lang sind, dass keines der `itemSets` mehr über `minSup` liegt. 

Der genaue Aufbau der Parameter kann den docstrings in den entsprechenden Methoden entnommen werden.


### Umbau
Um die Methoden des Repositories nutzen zu können haben wir uns in der Datei **db.py** eine Methode `getFromDB(maxPid)` geschrieben, welche abhängig von der `maxPid` eine Datenbank-Abfrage durchführt und damit das `trackSet` sowie das `playlistSet` generiert.
Diese werden so formatiert, dass sie als Input für `getAboveMinSup()` genutzt werden können.

Zudem haben wir uns eine Reihe von `print`-Befehlen in der Methode `aprioriFromDB()` ausgeben lassen, um feststellen zu können, warum das Programm nicht terminiert.
Die Ausgaben der `print`-Befehle werden im folgenden genauer analysiert. 

### Probleme
#### I - `getUnion()` liefert zu viele Kandidaten für `getAboveMinSup()`
Wie oben bereits beschrieben, iteriert `getUnion()` für jedes `trackSet` über alle `trackSets` und bildet davon ausgehend, alle möglichen Vereinigungsmengen.
Es wird lediglich geprüft, ob die Größe der Vereinigungsmenge `k` entspricht. 
Wir haben geprüft, wie groß der Output dieser Methode ist und festgestellt, dass die Größe der Ausgabe für `k = 2` genau der Anzahl aller zwei-elementigen Teilmengen über `trackSets` entspricht.

Dazu haben wir in der Datei **apriori_first.py** folgenden Befehl ausgeführt:
```
aprioriFromDB(maxPlaylists=2000, minSup=10, kMax=2)
```
Für 2000 Playlists und `minSup = 10` beträgt die Größe des Eingabeparameters `trackSets` 2188. Die Größe der Ausgabe (bestehend aus `trackSets` der Länge 2) beträgt 2.392.578.
Das entspricht genau der Menge an zwei-elementigen Teilmengen von 2188, welche sich mit der Binominalzahl berechnen lässt.
Die Rechnung dazu lautet: 2188 * 2187 / 2.
Das Problem ist nun, dass `getAboveMinSup()` eine Menge mit über 2 Mio. Elementen übergeben bekommt und für jedes Element über alle Playlists (in diesem Fall 2000) iterieren muss.
Das bedeutet, dass `getAboveMinSup()` bereits für 2000 Playlists (bei relativ hohem `minSup`) nicht terminiert.

#### II - `getUnion()` terminiert für größere Eingaben nicht
Nach der oben beschriebenen Komplexität werden für `trackSets` der Länge 2188, ca. 5 Mio. = 2188 * 2188   Iterationen durchgeführt.
Die Methode benötigt dazu auf unseren Computern ca. 15 Sekunden.
Setzen wir `minSup` runter auf 2 (was wir brauchen um möglichst viele Regeln zu bauen) muss `getUnion()` bereits mit 17349 `trackSets` rechnen und terminiert nicht, bzw. müsste ca. 300 Mio. Iterationen durchführen und dafür ca. 15 Minuten benötigen.
Um für die Challenge genug Regeln zu bilden, müssen wir jedoch mit ca. 100.000 `trackSets` rechnen und die Methode würde definitiv nicht terminieren (ca. 5 Mrd. Iterationen).  

#### III - Laufzeit von `getAboveMinSup()` steigt enorm mit wachsender Anzahl von Playlists
Wie oben beschrieben, ist die Komplexität von `getAboveMinSup()` nicht nur von den `trackSets` abhängig, sondern auch von der Anzahl der Playlists.
Dadurch bringt es uns nichts, die Ausgabe von `getUnion()` kleiner zu halten. Denn wenn über 1 Mio. Playlists iteriert werden muss, reichen 10.000 `trackSets` aus, damit die Methode nicht terminiert.

## Ansatz II - Dictionaries verwenden
Um die Probleme von Ansatz I lösen zu können, haben wir uns dazu entschieden den Apriori-Algorithmus selbst zu programmieren und für unseren Anwendungsfall zu optimieren.
Dieser Ansatz hat es uns letztendlich ermöglicht, die Assoziationsanalyse für 1 Mio. Playlists auf unseren Computern auszuführen, um so genug Regeln für die Challenge speichern zu können.
Der Ansatz kann in der Datei **apriori_spotify.py** begutachtet werden.

### Idee
Grundsätzlich war die Idee, nicht quadratisch über alle frequent ItemSets zu iterieren und stumpf alle möglichen Kombinationen zu erstellen, sondern nur Vereinigungen mit ItemSets zu bilden, die aus gleichen Playlists stammen. 
Es ergibt schließlich keinen Sinn, Vereinigungen mit ItemSets zu bilden, die aus disjunkten Playlists stammen und dann ohnehin im Folgeschritt durch `minSup` rausgefiltert werden.

Mit der Methode `printSupInfo('track_uri')` (s. Screenshots oben) haben wir gesehen, dass ein Track durchschnittlich in nur 28 Playlists vorkommt.
Es müssen also nicht für jeden Track Vereinigungen mit Tracks aus 1 Mio. Playlists gebildet werden, sondern es reicht Vereinigungen mit Tracks aus durchschnittlich 28 Playlists zu generierenn

Der grobe Aufbau um Kandidaten zu bilden sieht daher wie folgt aus:

```
for itemSet1 in frequentItemSets:
    for playlist in itemSet1.playlists:
        for itemSet2 in playlist.frequentItemSets:
            candidates.append(itemSet1.union(itemSet2))
```

Diese dreifache for-Schleife ersetzt die Methode `getUnion()`. Es gehen dabei keine Vereinigungen verloren, die im nächsten Schritt nicht sowieso entfernt worden wären.

Um von einem itemSet durch dessen Playlists zu iterieren und von einer Playlist durch deren itemSets zu iterieren, arbeiten wir mit zwei Dictionarys. Diese werden für jede Länge der itemSets neu erstellt und ermöglichen extrem schnelle Zugriffe.
Die Dictionaries haben die Bezeichnungen:
* `itemSet2Pids`: Ordnet jedem itemSet eine Menge von ***pids*** zu und hat drei Funktionen:
  1. Man kann für jedes itemSet über deren ***pids*** iterieren
  2. Man kann über die Länge der ***pids*** den Support eines itemSets abfragen: `len(pids) = sup`
  3. Über die Schnittmenge der ***pids*** lässt sich der Support von Vereinigungen berechnen (s. Foliensatz "Recommender Systeme" S. 29 - Erweiterung Apriori-TID)
* `pid2itemSets`: Ordnet jeder Playlist eine Menge von itemSets zu. 

Mit dem Dictionary `itemSet2Pids` kann direkt im Schleifendurchlauf der Support geprüft werden und die Methode `getAboveMinSup()` wird komplett hinfällig.
Die dreifache for-Schleife sieht dann wie folgt aus:

```
for itemSet1 in itemSet2Pids:
    for playlist in itemSet2Pids[itemSet1]:
        for itemSet2 in pids2ItemSets[playlist]:
            if itemSet2Pids[itemSet1].intersection(itemSet2Pids[itemSet2]) >= minSup:
                nextFrequentItemSets.append(itemSet1.union(itemSet2))
```

Die **Validierung** haben wir anhand des ersten Ansatzes und einer kleinen Datenmenge durchgeführt: 

**apriori_first.py**:
```
aprioriFromDB(maxPlaylists=300, minSup=2, kMax=2)
```
Nach ca. 85 Sekunden erscheint folgendes Output im Terminal: 

```
getUnion for k = 2 ... -> Done! 3376101 Candidates
getAboveMinSup for k = 2 ... -> Done! 19745 Tracks above minSup
```

**apriori_spotify.py**:
```
aprioriSpotify(item='track_uri', maxPid=300, minSup=2, kMax=2, b=-1, dbL1ItemSets=True)
```

Nach ca. 4 Sekunden erscheint folgendes Output im Terminal: 
```
k:  2 -> 19744 itemSets
```

**Fazit**: Deutlich schneller mit gleichem Ergebnis. Bzw. in diesem Beispiel ist 1 itemSet verloren gegangen. Für kleinere 
Eingaben ist die Anzahl jedoch exakt identisch.

Ein **Rechenbeispiel** verdeutlicht die Verbesserung:

Gegeben: 
* 1 Mio. frequentItemSets (diese entstehen, wenn die ***track_uris*** vom gesamten Datensatz auf `minSup = 2`geprüft werden)
* Durchschnittlich 28 Playlists in der ein Track bzw. eine ***track_uri*** vorkommt.
* Durschnittlich 25 frequent Tracks pro Playlists (Schätzung: Die durchschnittliche Playlist-Länge beträgt 66, ca. die Hälfte fällt durch `minSup = 2` weg und weitere Tracks fallen wegen Dopplung weg)

Rechnung:
* `getUnion()` führt 1 Mio * 1 Mio = 1 Billionen Iterationen durch und generiert 1.000.000 * 999.999 / 2 = ca. 500.000.000. 
  Danach führt `getAboveMinSup()` 500.000.000 * 1.000.000 Iterationen durch.
* Unsere Variante führt 1 Mio * 28 * 25 = 700 Mio. Iterationen durch.



### Implementierung
- Dictionarys
### Probleme
#### I - Es werden zu viele Regeln erstellen
#### II - Terminiert nicht für 1. Millionen Playlists
#### III - Datenbank-Abfrage dauert zu lange
### Lösungen
#### I
#### II
#### III
## Ansatz III - Library von mlxtend verwenden 
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

### Referenzen zu mlxtend
[1] http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/ \
[2] http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

## Recommendation
## Code-Struktur
* **preprocessingToCSV.py**
* **apriori_first.py**
* **apriori_spotify.py**
* **db.py**
* **helperMethods.py**
* **progressBar.py**
* **apriori_mlxtend.py**
* **recommendation.py**
* ***src/mpd_tools***
* ***challenge***
* ***data***
* ***data_processed***
* ***data_rules***

