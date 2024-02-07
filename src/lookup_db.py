import sqlite3
import numpy as np
import os

DATA_DIR = 'data'
DB_NAME = 'lookup_table_avg_centroids.db'

def get_db_location():
    # Path to the script being executed
    script_path = os.path.abspath(__file__)
    # Directory containing the script
    script_dir = os.path.dirname(script_path)
    # get the parent directory of the current directory
    parent_dir = os.path.dirname(script_dir)
    # add the data directory to the parent directory
    data_dir = os.path.join(parent_dir, DATA_DIR)
    db_location = os.path.join(data_dir, DB_NAME)
    return db_location
def create_database():
    # Connect to the SQLite database
    conn = sqlite3.connect(get_db_location())

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
    conn = sqlite3.connect(get_db_location())

    # Insert the data into the table
    conn.execute('INSERT INTO random_sets (block_size, num_entries, data) VALUES (?, ?, ?)',
                 (block_size, num_entries, serialized_data))
    conn.commit()
    conn.close()


def retrieve_sets():
    # Connect to the SQLite database
    conn = sqlite3.connect(get_db_location())

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

def vaccuum_db():
    conn = sqlite3.connect(get_db_location())
    conn.execute('VACUUM')
    conn.commit()
    conn.close()

