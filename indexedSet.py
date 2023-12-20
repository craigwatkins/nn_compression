from setIndex import SetIndex


class IndexedSet:
    """
        This class associates a set of vectors with a search index for faster nearest neighbor searches. It insulates
        itself from the particular method of indexing by using the SetIndex class.

        Properties: vectors: The set of vectors
                    set_index: The SetIndex object
                    block_size: The size of the tuple
    """

    def __init__(self, vectors, use_saved_index=False, index_type='annoy'):
        self.vectors = vectors
        self.set_index = None
        self.block_size = len(vectors[0])
        self.create_index(use_saved_index, index_type)

    def create_index(self, use_saved_index=False, index_type='annoy'):
        self.set_index = SetIndex(self.vectors, 0, index_type, use_saved_index=use_saved_index)
