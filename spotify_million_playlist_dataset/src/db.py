import psycopg2 as psycopg2
from enum import Enum


class _Query(Enum):
    MostPopularTracks = "SELECT track_uri, num_tracks " \
                        "FROM " \
                            "(SELECT track_uri, COUNT(track_uri) AS num_tracks " \
                            "FROM " \
                                "(SELECT track_uri " \
                                "FROM pcont " \
                                "GROUP BY track_uri, pid) AS x " \
                            "GROUP BY track_uri) AS y " \
                        "WHERE num_tracks >= 2 " \
                        "ORDER BY num_tracks DESC "
    NumPlaylists = "SELECT COUNT(pid) FROM playlists"
    NumTracks = "SELECT COUNT(id) FROM pcont"
    NumAvgPlaylistLength = "SELECT AVG(x.num_tracks) " \
                           "FROM (SELECT pid, COUNT(track_uri) AS num_tracks FROM pcont GROUP BY pid) AS x"
    NumUniqueTracks = "SELECT COUNT(track_uri) FROM tracks"
    NumUniqueTitles = "SELECT COUNT(track_name) FROM tracks"
    NumUniqueAlbums = "SELECT COUNT(album_uri) FROM albums"
    NumUniqueArtists = "SELECT COUNT(artist_uri) FROM artists"


def _dbReq(query):
    '''
    This method is used by all our db-requests. It is private, so that all sql-query's are collected in this file
    :param query: Can be a single sql-query or a list of query's for multiple db-requests with single connection
    :return: Returns the unfiltered output from the database
    '''
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="ldude001",
            user="ldude001",
            password="ldude001",
            port="9001",
        )
        cur = conn.cursor()

        print('PostgreSQL connected')
        if isinstance(query, list):  # in case of more than one query
            output = list()
            for subQuery in query:
                cur.execute(subQuery)
                output.append(cur.fetchall())
        else:
            cur.execute(query)
            output = cur.fetchall()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed')
    return output


def getFromDB(maxPlaylists):
    '''
    This method was used for our first apriori-attempt. We didn't use it anymore because it returns to much data
    and doesn't offer the possibility of quickly finding new candidates with the playlists of a track.
    :param maxPlaylists: Number of playlists that limits the load of data.
    :return:    Returns a set of all tracks from playlists with pid < maxPlaylists
                and a list of track-sets for each playlist
    '''
    playlists = []
    unique_tracks = set()
    playlistCounter = 0
    record = set()

    tracks = _dbReq(f'{"SELECT track_uri, pid FROM pcont WHERE pid<"}{maxPlaylists}')
    for track in tracks:
        if playlistCounter != track[1]:
            playlistCounter += 1
            playlists.append(record)
            record = set()
        track_uri = track[0]
        record.add(track_uri)
        unique_tracks.add(frozenset([track_uri]))
    playlists.append(record)
    return unique_tracks, playlists


def getL1PlaylistsDict(maxPlaylists, minSupPercent):
    '''
    This method generates the l1TrackSet by a database request, so the program doesn't need
    to work with tracks under the minSup. It saves memory and is faster than a method that filters the tracks.
    It can be used to iterate over the playlists of a track or getting the supply by calculate the length of the
    playlist-set.

    :param maxPlaylists:    Number of playlists that limits the load of data. Can be used for faster testing
                            or to free up the memory
    :param minSupPercent:   Specifies the minimum number of playlists the track must appear in
    :return:    A dictionary that assigns each track above minSup the playlists it appears in.
                The return value of the dictionary is of the form:
                {
                    frozenset({'track_uri'}): {pid, pid, ...},
                    frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'}): {0, 5},
                    frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}): {0, 5},
                    ...
                }
    '''
    minSup = maxPlaylists * minSupPercent   # the actual number of playlists the tracks must appears in
    l1PlaylistsDict = dict()     # the return value of this method
    resp = _dbReq("SELECT track_uri, support, pids "
                      "FROM "
                        "(SELECT track_uri, "
                                "COUNT(track_uri) AS support, "     # count in how many playlists the song appears
                                "string_agg(pid::character varying, ',') AS pids "    
                         "FROM "                                                     
                            "(SELECT track_uri, pid "    # some tracks are duplicated in a playlists
                            "FROM pcont "           # remove these duplicates with GROUP BY track_uri, pid 
                            "WHERE pid <" + str(maxPlaylists) +
                            " GROUP BY track_uri, pid) AS x "
                        "GROUP BY track_uri) AS y "
                      "WHERE support >=" + str(minSup) +
                      " ORDER BY support DESC")

    # formatting the response in a dictionary
    for line in resp:
        track_uri = frozenset([line[0]])  # track_uri of a track above minSup
        playlists = str(line[2])  # list of playlists in form of '5,1,0,12,...'
        l1PlaylistsDict[track_uri] = set(map(int, playlists.split(',')))



    return l1PlaylistsDict


def getL1TracksDict(maxPlaylists, minSupPercent):
    '''
    This method is used for a faster generation of candidates.

    :param maxPlaylists:    Number of playlists that limits the load of data. Can be used for faster testing
                            or to free up the memory
    :param minSupPercent:   Specifies the minimum number of playlists the track must appear in
    :return:    A dictionary that assigns each playlist the tracks above minSup.
                The return value of the dictionary is of the form:
                {
                    pid:    {frozenset({'track_uri', 'track_uri', ...}),
                            frozenset({'track_uri', 'track_uri', ...}), ...},
                    5:      {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1', ...}),
                            frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0'}), ...},
                    0:      {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1', ...}),
                            frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0'}), ...},
                    ...
                }
    '''
    minSup = maxPlaylists * minSupPercent   # the actual number of playlists the tracks must appears in
    l1TracksDict = dict()     # the return value of this method
    resp = _dbReq("SELECT pid, string_agg(x.track_uri, ',')"
                            " FROM "
                                "(SELECT track_uri, pid FROM pcont WHERE pid <" + str(maxPlaylists) + "GROUP BY track_uri, pid) AS x"
                                " INNER JOIN "
                                "(SELECT track_uri, COUNT(track_uri) AS support"
                                    " FROM "
                                        "(SELECT track_uri FROM pcont WHERE pid < " + str(maxPlaylists) + "GROUP BY track_uri, pid) AS z"
                                    " GROUP BY track_uri"
                                ") AS y"
                                " ON x.track_uri = y.track_uri"
                        " WHERE support >= " + str(minSup) +
                        " GROUP BY pid")
    # formatting the response in a dictionary
    for line in resp:
        pid = line[0]  # pid of playlist
        tracks = str(line[1]).split(',')  # line[1] in form 'track_uri_1,track_uri_2,...'
        l1TracksDict[pid] = set()
        for track in tracks:
            l1TracksDict[pid].add(frozenset([track]))

    return l1TracksDict


def dbValidate():
    '''
    This method prints infos about our database that can be compared with the project stats
    :return:    Doesnt return anything. Only prints data.
    '''
    resp = _dbReq([_Query.NumPlaylists.value,
                   _Query.NumTracks.value,
                   _Query.NumUniqueTracks.value,
                   _Query.NumUniqueAlbums.value,
                   _Query.NumUniqueArtists.value,
                   _Query.NumUniqueTitles.value,
                   _Query.NumAvgPlaylistLength.value])
    print("number of playlists", str(resp[0])[2:-3])
    print("number of tracks", str(resp[1])[2:-3])
    print("number of unique tracks", str(resp[2])[2:-3])
    print("number of unique albums", str(resp[3])[2:-3])
    print("number of unique artists", str(resp[4])[2:-3])
    print("number of unique titles", str(resp[5])[2:-3])
    print("avg playlist length", str(resp[6])[11:-5])




