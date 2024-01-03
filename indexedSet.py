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

    def get_matches(self, vectors):
        """
        This method returns the closest match for each vector in vectors
        :param vectors: The vectors to search for
        :return: The closest match for each vector, and the distance to that match
        """
        distances, vector_indices = self.tree.query(vectors)
        return distances, vector_indices


