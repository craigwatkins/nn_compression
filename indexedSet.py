from scipy.spatial import KDTree
from annoy import AnnoyIndex
import os

class IndexedSet:
    """
        This class associates a set of vectors with a search index for faster nearest neighbor searches. It insulates
        itself from the particular method of indexing by using the SetIndex class.

        Properties: vectors: The set of vectors
                    set_index: The SetIndex object
                    block_size: The size of the tuple
    """

    def __init__(self, vectors, make_index=True):
        self.vectors = vectors
        self.set_index = None
        self.block_size = len(vectors[0])
        self.tree = KDTree(vectors)
        self.match_dict = {}
        self.match_dict_reverse = {}
        self.file_name = ''
        self.file_name = f'''annoyIndexes/tree_{len(self.vectors)}.ann'''
        self.annoy_num_trees = 6
        self.index = None
        if make_index:
            self.create_index()
        else:
            self.load_index()

    def get_matches(self, vectors):
        """
        This method returns the closest match for each vector in vectors
        :param vectors: The vectors to search for
        :return: The closest match for each vector, and the distance to that match
        """
        index, distance = self.index.get_nns_by_vector(vectors, 1, include_distances=True)
        return index[0], distance[0]

    def create_index(self):
        self.index = AnnoyIndex(self.block_size, metric='euclidean')
        for i, vector in enumerate(self.vectors):
            self.index.add_item(i, vector)
        self.index.build(self.annoy_num_trees)
        #self.index.save(self.file_name)

    def load_index(self):
        if os.path.isfile(self.file_name):
            self.index = AnnoyIndex(self.block_size, 'euclidean')
            self.index.load(self.file_name)


    def make_match_dict(self, super_dict):
        """
        This method creates a dictionary of matches for each vector in the set that interfaces with the super_dict
        :return: A dictionary of matches
        """
        for i, vector in enumerate(self.vectors):
            super_index = super_dict[tuple(vector)]
            self.match_dict[tuple(vector)] = super_index
            self.match_dict_reverse[super_index] = tuple(vector)
