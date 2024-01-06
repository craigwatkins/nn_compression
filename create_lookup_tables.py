from lookupTable import LookupTable
import lookup_db
from indexedSet import IndexedSet


def create_lookup_table():
    sample_sets = lookup_db.retrieve_sets()
    # remove the last elements of the last set to make room for special values
    sample_sets[-1] = sample_sets[-1][:-3]
    index_sets = [IndexedSet(x) for x in sample_sets]
    lookup_table_1 = LookupTable(index_sets)
    print("lookup table max index", lookup_table_1.max_index)
    return lookup_table_1
