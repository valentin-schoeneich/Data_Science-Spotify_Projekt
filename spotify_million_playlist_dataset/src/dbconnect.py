import psycopg2 as psycopg2
try:
    conn = psycopg2.connect(
        host="localhost",
        database="ldude001",
        user="ldude001",
        password="ldude001",
        port="9001",
    )
    cur = conn.cursor()
    print('PostgreSQL database version:')
    cur.execute('SELECT * FROM albums')
    print(cur.fetchall())
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
        print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')