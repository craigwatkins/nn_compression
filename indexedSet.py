from scipy.spatial import KDTree


class IndexedSet:
    """
        This class associates a set of vectors with a search index for faster nearest neighbor searches. It insulates
        itself from the particular method of indexing by using the SetIndex class.

        Properties: vectors: The set of vectors
                    set_index: The SetIndex object
                    block_size: The size of the tuple
    """

    def __init__(self, vectors):
        self.vectors = vectors
        self.set_index = None
        self.block_size = len(vectors[0])
        self.tree = KDTree(vectors)
        self.match_dict = {}
        self.match_dict_reverse = {}

    def get_matches(self, vectors):
        """
        This method returns the closest match for each vector in vectors
        :param vectors: The vectors to search for
        :return: The closest match for each vector, and the distance to that match
        """
        distances, vector_indices = self.tree.query(vectors)
        return distances, vector_indices

    def make_match_dict(self, super_dict):
        """
        This method creates a dictionary of matches for each vector in the set
        :return: A dictionary of matches
        """
        for i, vector in enumerate(self.vectors):
            super_index = super_dict[tuple(vector)]
            self.match_dict[tuple(vector)] = super_index
            self.match_dict_reverse[super_index] = tuple(vector)



