from lookupTable import LookupTable
import lookup_db
from indexedSet import IndexedSet


def create_lookup_table():
    lookup_list = []
    """
    sample_set = db.get_samples(2)
    sample_set = [tuple(x) for x in sample_set]
    num_samples = 2 ** 17
    # define set with block_size, num_samples, max_palette_bit_size, sample_set
    # lookup_list.append(RandomSet(8, num_samples, sample_set))
    # lookup_list.append(RandomSet(7, num_samples, sample_set))
    lookup_list.append(RandomSet(6, num_samples*2, sample_set))
    # lookup_list.append(RandomSet(5, num_samples, sample_set))
    lookup_list.append(RandomSet(4, num_samples*2, sample_set))
    lookup_list.append(RandomSet(3, num_samples*2, sample_set))
    lookup_list.append(RandomSet(2, num_samples, sample_set))
    lookup_list.append(RandomSet(1, num_samples, sample_set))
    sorted_lists = []
    for lookup in lookup_list:
        samples = lookup.values
        # sort by distance from origin
        sorted_samples = sorted(samples, key=lambda x: np.linalg.norm(x))
        lookup_db.insert_set(sorted_samples)
    """
    print("making lookup table")
    use_saved_index = True
    sample_sets = lookup_db.retrieve_sets()
    index_sets = [IndexedSet(x, use_saved_index=use_saved_index) for x in sample_sets]
    lookup_table_1 = LookupTable(index_sets, use_saved_indexes=True)
    return lookup_table_1

