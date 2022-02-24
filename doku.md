# <center> Data Science WS 2021/2022 <br/> Dokumentation </center>

## <center> Lösung der AI crowd Spotify Million Playlist Dataset Challenge mittels Assoziationsanalyse </center>

## Ziel
Ziel ist es mit einer Assoziationsanalyse Regeln für möglichst viele Tracks zu generieren, mithilfe derer dann 
die Recommendation gebaut werden kann. Eine Regel soll dabei folgende Form haben: 
`[{A, ...}, {B, ...}, Konfidenz]` bzw. `[antecedent, consequent, confidence]`  
Die Menge an Stelle 0 der Liste besteht aus den Vorgängern (im Code `antecedent`), Stelle 1 aus den Nachfolgern 
(im Code `consequent`).
Die Konfidenz an Stelle 2 der Liste steht für die Wahrscheinlichkeit, dass die
Nachfolger eintreten, gegeben der Wahrscheinlichkeit, dass die Vorgänger bereits eingetroffen sind.  
In den meisten Fällen bestehen beide Mengen aus nur einem Element. Eine Regel könnte dann lauten: `[{A}, {B}, 0.5]`.
In einem solchen Fall kann die Regel wie folgt gelesen werden:  
"Wenn der Track A in der Playlist enthalten ist, wird zu 50% auch Track B enthalten sein."


### Ansätze für Predictions
## Ansatz I - Vorhandenes Repository umbauen
### Funktionsweise
### Problem
## Ansatz II - Dictionaries verwenden
### Funktionsweise
### Probleme
#### I - Es werden zu viele Regeln erstellen
#### II - Terminiert nicht für 1. Millionen Playlists
#### III - Datenbank-Abfrage dauert zu lange
### Lösungen
#### I
#### II
#### III
## Ansatz III - Library von mlxtend verwenden 
## Technische Dokumentation
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

