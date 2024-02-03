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


def common_diffs(centroids, start_index, dict_size, db_name, samples_per_image, sample_size=1):
    samples = sample_diffs(db_name, num_images=8090, num_samples=samples_per_image, sample_size=sample_size)
    samples = samples.reshape(-1, 3)  # flatten the last dimension
    from scipy.spatial import KDTree
    tree = KDTree(centroids)
    distances, vector_indices = tree.query(samples)
    grouped_vector_indices = [tuple(vector_indices[i:i+sample_size]) for i in range(0, len(vector_indices), sample_size)]
    # get the number of occurrences of each group of vectors
    from collections import Counter
    counts = Counter(grouped_vector_indices)
    # sort the counts by the number of occurrences
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # get the most common pairs
    most_common = sorted_counts[:dict_size]
    most_common = [x[0] for x in most_common]
    most_common_dict = {x: i + start_index for i, x in enumerate(most_common)}
    return most_common_dict
def get_centroids(k, samples):

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(samples)
    # Calculate centroids
    centroids = kmeans.cluster_centers_
    return centroids

def get_row_distributions(pixel_centroids, num_row_centroids, db_name, samples_per_image, sample_size=1):
    samples = sample_diffs(db_name, num_images=8090, num_samples=samples_per_image, sample_size=sample_size)
    samples = samples.reshape(-1, 3)  # remove the rows
    from scipy.spatial import KDTree
    # replace each pixel with matching centroid
    tree = KDTree(pixel_centroids)
    distances, vector_indices = tree.query(samples)
    # group the indexes of the centroids into rows
    grouped_vector_indices = [tuple(vector_indices[i:i + sample_size]) for i in
                              range(0, len(vector_indices), sample_size)]
    # for each entry in grouped_vector_indexes (each row), get the number of occurrences of vector_indices within it
    from collections import Counter
    row_distributions = []
    print('getting row distributions')
    for i, group in enumerate(grouped_vector_indices):
        group_counter = Counter(group)
        centroid_table = [0] * len(pixel_centroids)
        # sort group_counter keys smallest to largest
        sorted_keys = sorted(group_counter.keys())
        for j, key in enumerate(sorted_keys):
            value = group_counter[key]
            centroid_table[key] = value/sample_size
        row_distributions.append(tuple(centroid_table))
    print('getting row centroids')
    row_centroids = get_centroids(num_row_centroids, np.array(row_distributions))

    return row_centroids



def create_lookup_table():

    sample_sets = lookup_db.retrieve_sets()
    # sort the sets in sample sets by length, shortest first
    sample_sets.sort(key=lambda x: len(x))

    start_1 = len(sample_sets[4]) + 1
    most_common_dict = {}
    #most_common_dict = common_diffs(sample_sets[4], start_1, 100, 'image_avg_diffs.db', 1, 2)
    #row_centroids = get_row_distributions(sample_sets[3], 4, 'image_avg_diffs.db', 4, 125)

    row_centroids = []
    """
    start_2 = start_1 + len(most_common_dict) + 1
    most_common_dict2 = common_diffs(sample_sets[0], start_2, 50, 'image_avg_diffs.db', 100, 3)
    start_3 = start_2 + len(most_common_dict2) + 1
    most_common_dict3 = common_diffs(sample_sets[0], start_3, 20, 'image_avg_diffs.db', 100, 4)
    start_4 = start_3 + len(most_common_dict3) + 1
    most_common_dict4 = common_diffs(sample_sets[0], start_4, 20, 'image_avg_diffs.db', 100, 5)
    start_5 = start_4 + len(most_common_dict4) + 1
    most_common_dict5 = common_diffs(sample_sets[0], start_5, 10, 'image_avg_diffs.db', 100, 6)

    most_common_dict.update(most_common_dict2)
    most_common_dict.update(most_common_dict3)
    most_common_dict.update(most_common_dict4)
    most_common_dict.update(most_common_dict5)
    """
    #print("most common dict length", len(most_common_dict))

    #most_common_dict = most_common_dict2


    # sample_sets = [sample_sets[5], sample_sets[-1]]
    sample_sets = [sample_sets[6]]
    index_sets = [IndexedSet(x) for x in sample_sets]
    lookup_table_1 = LookupTable(index_sets)
    #print("lookup table max index", lookup_table_1.max_index)



    return lookup_table_1, most_common_dict, row_centroids

# create_lookup_table()
