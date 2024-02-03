import sqlite3
import numpy as np


def create_database():
    # Connect to the SQLite database
    conn = sqlite3.connect('lookup_table_avg_centroids.db')

    # Create a new SQLite table
    conn.execute('''CREATE TABLE random_sets
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     block_size INTEGER NOT NULL,
                     num_entries INTEGER NOT NULL,
                     data BLOB NOT NULL)''')
    conn.commit()
    conn.close()


def insert_set(data):
    # Convert data to a 2D numpy array of type int16
    block_size = len(data[0])
    num_entries = len(data)
    np_data = np.array(data, dtype=np.int16)

    # Serialize the numpy array to bytes
    serialized_data = sqlite3.Binary(np_data.tobytes())

    # Connect to the SQLite database
    conn = sqlite3.connect('lookup_table.db')

    # Insert the data into the table
    conn.execute('INSERT INTO random_sets (block_size, num_entries, data) VALUES (?, ?, ?)',
                 (block_size, num_entries, serialized_data))
    conn.commit()
    conn.close()


def retrieve_sets():
    # Connect to the SQLite database
    conn = sqlite3.connect('lookup_table_avg_centroids.db')

    # Retrieve all rows from the table
    cursor = conn.execute('SELECT block_size, num_entries, data FROM random_sets')
    rows = cursor.fetchall()

    # Process each row
    result = []
    for block_size, num_entries, data in rows:
        # Convert bytes back to a numpy array
        np_data = np.frombuffer(data, dtype=np.int16).reshape(num_entries, block_size)
        # Convert numpy array to a list of tuples
        list_of_tuples = [tuple(row) for row in np_data]
        result.append(list_of_tuples)

    conn.close()
    return result




