class LookupTable:
    """
    Class that represents a lookup table. It uses a collection of sets to create a lookup table for vectors.
    Parameters:
        set_list: The list of sets that will be used to create the lookup table.
        use_saved_indexes: A boolean that determines whether to use saved search indexes or not.
    """
    def __init__(self, set_list=[], use_saved_indexes=False):
        self.set_list = set_list
        self.max_block_size = max([x.block_size for x in self.set_list])
        self.index_dict = {}
        self.index_dict_reverse = {}
        self.palette = []
        self.max_index = 0
        self.sorted_sets = []
        self.create_lookup_table()

    def create_lookup_table(self):
        super_set = [[0], [1], [2]]
        for i, a_set in enumerate(self.set_list):
            super_set += a_set.vectors
        super_set = [tuple(x) for x in super_set]
        self.index_dict = {value: index for index, value in enumerate(super_set)}
        self.index_dict_reverse = {index: value for index, value in enumerate(super_set)}
        self.palette = super_set
        self.max_index = len(super_set)
