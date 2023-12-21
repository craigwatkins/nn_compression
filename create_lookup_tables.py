from lookupTable import LookupTable
import lookup_db
from indexedSet import IndexedSet


def create_lookup_table():
    print("making lookup table")
    use_saved_index = True
    sample_sets = lookup_db.retrieve_sets()
    # remove the last elements of the last set to make room for special values
    sample_sets[-1] = sample_sets[-1][:-3]
    index_sets = [IndexedSet(x, use_saved_index=use_saved_index) for x in sample_sets]
    lookup_table_1 = LookupTable(index_sets, use_saved_indexes=use_saved_index)
    print("lookup table max index", lookup_table_1.max_index)
    return lookup_table_1
