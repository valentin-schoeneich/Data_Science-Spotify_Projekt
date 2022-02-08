import psycopg2 as psycopg2


def _dbReq(query):
    """
    This method is used by all our db-requests. It is private, so that all sql-query's are collected in this file
    :param query: Can be a single sql-query or a list of query's for multiple db-requests with single connection
    :return: Returns the unfiltered output from the database
    """
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


def _getQuerySupportOfItems(item, maxPid):
    """
    The output of the query (not this method) is the support and a list of playlists for each item from a playlist
    with pid < maxPid.
    :param item:    Could be either 'track_uri', 'track_name', 'artist_uri', 'album_uri' or 'name' (name of playlist)
    :param maxPid:  Maximum size of playlists for where-clause
    :return:        String, that can be used for a sql-query
    """
    _validateItem('getQuerySupportOfItems', item)
    query = f"SELECT {item}, COUNT({item}) AS support, string_agg(pid::character varying, ',') AS pids " \
            f"FROM ({_getQueryUniqueItemsOfPlaylists(item, maxPid)}) AS x GROUP BY {item}"
    return query


def _getQueryUniqueItemsOfPlaylists(item, maxPid):
    """
    The output of the query (not this method) is a list of (item, pid)-pairs. The query can be used to calculate the
    support in a further query without counting duplicates inside a playlist.
    :param item:    Could be either 'track_uri', 'track_name', 'artist_uri', 'album_uri' or 'name' (name of playlist)
    :param maxPid:  Maximum size of playlists for where-clause
    :return:        String, that can be used for a sql-query
    """
    _validateItem('getQueryUniqueItemsOfPlaylists', item)
    if item == 'track_uri':
        query = f"SELECT {item}, pid FROM pcont WHERE pid < {maxPid} GROUP BY {item}, pid"
    elif item == 'name':
        query = f"SELECT {item}, pid FROM playlists WHERE pid < {maxPid} GROUP BY {item}, pid"
    else:
        query = f"SELECT {item}, pid FROM pcont NATURAL JOIN tracks WHERE pid < {maxPid} GROUP BY {item}, pid"
    return query


def _validateItem(name, item):
    validItem = {'track_uri', 'track_name', 'artist_uri', 'album_uri', 'name'}
    if item not in validItem:
        raise ValueError(name, ": item must be one of %r." % validItem)


def getFromDB(maxPid):
    """
    This method was used for our first apriori-attempt. We didn't use it anymore because it returns to much data
    and doesn't offer the possibility of quickly finding new candidates with the playlists of a track.
    :param      maxPid: Number of playlists that limits the load of data.
    :return:    Returns unique tracks of each playlist and unique tracks of all playlist
    """
    playlists = []
    unique_tracks = set()
    playlistCounter = 0
    record = set()

    tracks = _dbReq(f'{"SELECT track_uri, pid FROM pcont WHERE pid<"}{maxPid}')
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


def getL1ItemSet2Pids(item, maxPid, minSupPercent):
    """
    This method generates the l1ItemSet by a database request, so the program doesn't need
    to work with items under the minSup. It saves memory and is faster than a method that filters the items only with
    maxPid.
    It can be used to iterate over the playlists of a itemSet or getting the supply by calculate the length of the
    pid-set.
    :param item:            Could be either 'track_uri', 'track_name', 'artist_uri', 'album_uri'
                            or 'name' (name of playlist)
    :param maxPid:          Number of playlists that limits the load of data. Can be used for faster testing
                            or to free up the memory
    :param minSupPercent:   Specifies the minimum number of playlists the track must appear in
    :return:    A dictionary that lists for each itemSet above minSup the playlists it appears in. In this case
                of l1ItemSet, every itemSet have the same length = 1
                The return value of the dictionary is of the form:
                {
                    frozenset({'item'}): {pid, pid, ...},
                    ...
                }
                e.g.
                {
                    frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'}): {0, 5, 2},
                    frozenset({'spotify:track:0XUfyU2QviPAs6bxSpXYG4'}): {1, 3, 5}
                }
    """
    _validateItem('getL1ItemSet2Pids', item)
    minSup = maxPid * minSupPercent  # the actual number of playlists the item must appears in
    l1ItemSet2Pids = dict()  # the return value of this method
    resp = _dbReq(f"SELECT {item}, support, pids "
                  f"FROM ({_getQuerySupportOfItems(item, maxPid)}) AS x "
                  f"WHERE support >= {minSup}")

    # formatting the response in a dictionary
    for line in resp:
        item = frozenset([line[0]])
        pids = str(line[2])  # list of playlists in form of '5,1,0,12,...'
        l1ItemSet2Pids[item] = set(map(int, pids.split(',')))

    return l1ItemSet2Pids


def getL1Pid2ItemSets(item, maxPid, minSupPercent):
    """
    This method is used for a faster generation of candidates.

    :param item:            Could be either 'track_uri', 'track_name', 'artist_uri', 'album_uri'
                            or 'name' (name of playlist)
    :param maxPid:          Number of playlists that limits the load of data. Can be used for faster testing
                            or to free up the memory
    :param minSupPercent:   Specifies the minimum number of playlists the item must appear in
    :return:    A dictionary that lists for each playlist the itemSets above minSup. In this case of l1ItemSet,
                every itemSet have the same length = 1
                The return value of the dictionary is of the form:
                {
                    pid:    {frozenset({'item'}), frozenset({'item'}), ...},
                    ...
                }
                e.g
                {
                    5:      {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'}),
                            frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0'})},
                    0:      {frozenset({'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1'}),
                            frozenset({'spotify:track:3H1LCvO3fVsK2HPguhbml0'})},
                }
    """
    _validateItem('getL1Pid2ItemSets', item)
    minSup = maxPid * minSupPercent  # the actual number of playlists the item must appears in
    l1Pid2ItemSets = dict()  # the return value of this method
    resp = _dbReq(f"SELECT pid, string_agg(x.{item}, ',') "
                  f"FROM ({_getQueryUniqueItemsOfPlaylists(item, maxPid)}) AS x "
                  f"NATURAL JOIN ({_getQuerySupportOfItems(item, maxPid)}) AS y "
                  f"WHERE support >= {minSup} "
                  f"GROUP BY pid")
    # formatting the response in a dictionary
    for line in resp:
        pid = line[0]  # pid of playlist
        items = str(line[1]).split(',')  # line[1] in form 'item1,item2,...'
        l1Pid2ItemSets[pid] = set()
        for item in items:
            l1Pid2ItemSets[pid].add(frozenset([item]))

    return l1Pid2ItemSets


def getNumUniqueItems(item, maxPid):
    """
    To calculate the percentage of items for which a rule was created, you have to know how many items exists in total.
    This method helps for this calculation by giving the total number of unique appearances of a item
    depended on maxPid.
    :param item:    Could be either 'track_uri', 'track_name', 'artist_uri', 'album_uri' or 'name' (name of playlist)
    :param maxPid:  Maximum size of playlists for where-clause
    :return:    Returns the number of appearances of a item in a range given by maxPid. The appearances are counted
                unique over all playlists with pid < maxPid
    """
    _validateItem('getNumUniqueItems', item)
    if item == 'track_uri':
        resp = _dbReq(f"SELECT COUNT({item}) "
                      f"FROM (SELECT {item} FROM pcont WHERE pid < {maxPid} GROUP BY {item}) AS x")
    elif item == 'name':
        resp = _dbReq(f"SELECT COUNT({item}) "
                      f"FROM (SELECT {item} FROM playlists WHERE pid < {maxPid} GROUP BY {item}) AS x")
    else:
        resp = _dbReq(f"SELECT COUNT({item}) "
                      f"FROM (SELECT {item} FROM pcont NATURAL JOIN tracks WHERE pid < {maxPid} GROUP BY {item}) AS x")
    return int(str(resp)[2:-3])


def dbValidate():
    """
    This method prints infos about our database that can be compared with the project stats
    :return:    Doesnt return anything. Only prints data.
    """
    NumUniqueTracks = "SELECT COUNT(track_uri) FROM tracks"
    NumUniqueTitles = "SELECT COUNT(name) FROM (SELECT name FROM playlists GROUP BY(name)) AS x"
    NumUniqueAlbums = "SELECT COUNT(album_uri) FROM albums"
    NumUniqueArtists = "SELECT COUNT(artist_uri) FROM artists"
    NumPlaylists = "SELECT COUNT(pid) FROM playlists"
    NumTracks = "SELECT COUNT(id) FROM pcont"
    resp = _dbReq([NumPlaylists, NumTracks, NumUniqueTracks, NumUniqueAlbums, NumUniqueArtists, NumUniqueTitles])
    print("number of playlists", str(resp[0])[2:-3])
    print("number of tracks", str(resp[1])[2:-3])
    print("number of unique tracks", str(resp[2])[2:-3])
    print("number of unique albums", str(resp[3])[2:-3])
    print("number of unique artists", str(resp[4])[2:-3])
    print("number of unique titles", str(resp[5])[2:-3])


