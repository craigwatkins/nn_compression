from annoy import AnnoyIndex
import numpy as np
import os
import math


class SetIndex:
    """
    This class uses the ANNOY library to create a search index for a set of vectors.
    Properties:
        sorted_values: list of tuples sorted by distance from null vector
        set_list_index: int, index of set in lookup table
        index_type: string, type of index to use
        index_offset: int, offset for index
    Methods:
        __init__: initialize SetIndex object
        create_index: creates index for set
        get_closest_match: returns closest match to a given vector
    """
    def __init__(self, sorted_values, set_list_index, index_type='annoy', annoy_num_trees=30, index_offset=0, use_saved_index=False):
        # convert sorted_values to numpy array
        self.sorted_values = np.array(sorted_values)
        self.block_size = len(sorted_values[0])
        self.index_type = index_type
        self.index_offset = index_offset
        self.annoy_num_trees = annoy_num_trees
        self.max_bit_size = int(math.ceil(math.log2(len(sorted_values))))
        self.set_list_index = set_list_index
        self.use_saved_index = use_saved_index
        self.index = self.create_index()

    def create_index(self):
        # create index
        index = None
        if self.index_type == 'annoy':
            file_name = f'''annoyIndexes/tree_{self.max_bit_size}_{self.annoy_num_trees}_{self.block_size}_{self.set_list_index}.ann'''
            # check to see if the tree has already been built
            if os.path.isfile(file_name) and self.use_saved_index:
                index = AnnoyIndex(self.block_size, 'euclidean')
                try:
                    index.load(file_name)
                except OSError as e:
                    print(f"An error occurred: {e}.")
            else:
                index = AnnoyIndex(self.block_size, metric='euclidean')
                for i, vector in enumerate(self.sorted_values):
                    index.add_item(i, vector)
                index.build(self.annoy_num_trees)
                try:
                    index.save(file_name)
                except OSError as e:
                    print(f"An error occurred: {e}. Annoy file is probably already in use.")
        return index

    def get_closest_match(self, vector):
        if self.index_type == 'annoy':
            index = self.index.get_nns_by_vector(vector, 1)
            return self.sorted_values[index[0]]

