from lookupTable import LookupTable
import lookup_db
from indexedSet import IndexedSet
import numpy as np
from sklearn.cluster import KMeans
SEED = 42


def sample_diffs(db_name, num_images, num_samples, sample_size=1):
    import sqlite3
    import random
    import numpy as np
    height = 127
    width = 127
    # set random seed
    random.seed(SEED)
    # Connect to the database
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Retrieve the diffs for a specified number of images
    cursor.execute("SELECT diffs FROM images LIMIT ?", (num_images,))
    rows = cursor.fetchall()

    all_samples = []

    for row in rows:
        # Reshape the diff data to its original shape
        diff_array = np.frombuffer(row[0], dtype=np.int16).reshape(height, width, 3)

        for _ in range(num_samples):
            # Randomly select a row
            row_idx = random.randint(0, height-1)
            # Ensure sample doesn't cross rows
            if 128 - sample_size > 0:
                start_col_idx = random.randint(0, width - sample_size)
                sample = diff_array[row_idx, start_col_idx:start_col_idx + sample_size, :]
                all_samples.append(sample)

    connection.close()
    return np.array(all_samples)


def create_lookup_table():
    sample_sets = lookup_db.retrieve_sets()
    # sort the sets in sample sets by length, shortest first
    sample_sets.sort(key=lambda x: len(x))
    sample_sets = [sample_sets[6], sample_sets[-1]]
    # remove duplicates from largest set
    sample_sets[-1] = list(set(sample_sets[-1]) - set(sample_sets[0]))
    index_sets = [IndexedSet(x) for x in sample_sets]
    lookup_table_1 = LookupTable(index_sets)
    #print("lookup table max index", lookup_table_1.max_index)
    return lookup_table_1
