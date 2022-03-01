# The Apriori-Algorithm 

## Getting Started

### Set-up connection to Database

Before start, we have to set up an SSH-Tunnel to our database.   
In our case use the command:
```
psql -h localhost -p 9001 -U ldude001 -d ldude001
```

To prove the connection and verify the consistency of the database, call in the file **db.py** the following command:

```
dbValidate()
```

The output in terminal should be this:

```
PostgreSQL connected
Database connection closed
number of playlists 1000000
number of tracks 66346428
number of unique tracks 2262292
number of unique albums 734684
number of unique artists 295860
number of unique titles 92944
```

### Check paths

In file **helperMethods.py** we defined an Enum with four paths. The paths are used for following procedures:
* `pathToData` - Path to the 1000 JSON-Files from AIcrowd  
  Is used in file **preprocessingToCSV.py** to create CSV-Files.
* `pathToProcessedData` - Path to CSV-Files created by preprocessing  
  Is used in file **preprocessingToCSV.py** to save the preprocessed data.  
  Is used in file **apriori_spotify.py** to load the preprocessed data.
* `pathToRules`  
  Is used in file **apriori_spotify.py** to save the calculated rules.
  Is used in file **recommendation.py** to load the calculated rules.
* `pathToChallenge`  
  Is used in file **recommendation.py** to load the **challenge_set.json** from AIcrowd and to save the calculated submission.
  
The easiest way is probably to adopt our project-structure. Therefore, create folder and copy files in it, that it looks like:

* spotify_million_playlist_dataset/  
  * challenge/
    * challenge_set.json
  * data/ 
    * mpd.slice.0-999.json
    * mpd.slice.1000-1999.json
    * ...
  * data_processed/
  * data_rules/
  * src/

Alternatively you can adapt the paths to existing folders. 

### How to use

There are some dependencies in our methods where you need to create a CSV-File first. If you only want to do a single method-call look at following dependencies.
For a full recommendation take a look at the next chapter.

1. `savePopularTracks(maxFiles)` in **preprocessingToCSV.py**   
   Following files in **data_processed/** required:  
   * artist_uri2track_uris_{**maxFiles**}.csv  
   * album_uri2track_uris_{**maxFiles**}.csv
   * track_uri2pids_{**maxFiles**}.csv
   
   Therefore, call:
   ```
   csvItem2Values(maxFiles=maxFiles, keys={'album_uri', 'artist_uri'},  values='track_uri', minSup=1)
   csvItem2Values(maxFiles=maxFiles, keys='track_uri', values='pid', minSup=1)
   ``` 
2. `aprioriSpotify(item, maxPid, dbL1ItemSets=False)` in **apriori_spotify.py**  
     
    Following file in **data_processed/** required:  
    -  {**item**}2pids_{**maxFiles**}.csv

    Therefore, call:
    ```
    csvItem2Values(maxFiles=maxFiles, keys={item}, values='pid', minSup=1)
    ```

   **Notes:**  
   - If `dbL1ItemSets=True` this method will load the data from our database. No preceding method-call required.
   - Remind that `dbL1ItemSets=True` won't be work for the whole data
     We recommend to set `dbL1ItemSets=True` if you use less than 10.000 Playlists and set `dbL1ItemSets=False` if you want to use more than 100.000 Playlists. 
     For all between it depends on how often you want to call the method and your endurance. It will take 1-8 minutes.
   - It is advisable to choose `maxpid` in `csvItem2Values()` like in `apriori_spotify()` to keep the loading time as short as possible. Alternatively you can choose `maxpid=1000`. This will also work for `maxpid` < 1000.  
3. `aprioriPname(consequents, maxPid)` in **apriori_spotify.py**  
    Following file in **data_processed/** required: 
    - name2Pids_{**maxFiles**}.csv   
    - name2{**consequents**}_{**maxFiles**}.csv  
    - {**consequents**}2Pids_{**maxFiles**}.csv
    
    Therefore, call:
    ```
    csvItem2Values(maxFiles=maxFiles, keys='name', values={{consequents}, 'pid'}, minSup=1)
    csvItem2Values(maxFiles=maxFiles, keys={consequents}, values='pid', minSup=1)
    ```
4.  `getL1ItemSet2ValuesFromCSV(item, value, maxFiles)` in **helperMethods.py**  
    Following file in **data_processed/** required:
    - {**item**}2{**value**}_{**maxFiles**}.csv
    
    Therefore, call:
    ```
    csvItem2Values(maxFiles=maxFiles, keys={item}, values={value}, minSup=1)
    ```
5.  `printSupInfo(item)` in **helperMethods.py**    
    Following file in **data_processed/** required:
    - {**item**}2pids_1000.csv
    
    Therefore, call:
    ```
    csvItem2Values(maxFiles=1000, keys={item}, values=pids, minSup=1)
    ```
6.  `recommendation()` in **recommendation.py**  
    Works only for whole data. Take a look at next chapter.
   

### Make recommendation over whole data
The following steps will show you how to make a recommendation over the whole data.
For this calculation 32 GB of RAM is required. It will take 3 to 6 hours.

1. **Preprocessing in file preprocessingToCSV.py**  
   ```
   createFiles()
   ``` 
2. **Create Rules in file apriori_spotify.py**
   ```
   aprioriSpotify(item='track_uri', maxPid=10000, minSup=2, kMax=2, b=10, p=10, dbL1ItemSets=False, saveRules=True)
   aprioriSpotify(item='album_uri', maxPid=10000, minSup=2, kMax=2, b=10, p=10, dbL1ItemSets=False, saveRules=True)
   aprioriSpotify(item='artist_uri', maxPid=10000, minSup=2, kMax=2, b=10, p=10, dbL1ItemSets=False, saveRules=True)
   aprioriPname(consequents='track_uri', maxPid=1000, minSup=2, minConf=0.2, saveRules=True)
   aprioriPname(consequents='album_uri', maxPid=1000, minSup=2, minConf=0.2, saveRules=True)
   aprioriPname(consequents='artist_uri', maxPid=1000, minSup=2, minConf=0.2, saveRules=True)
   ```
3. **Make recommendation in recommendation.py**
   ```
   recommendation()
   ``` 
