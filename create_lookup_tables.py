from lookupTable import LookupTable
import lookup_db
from indexedSet import IndexedSet


def create_lookup_table():
    print("making lookup table")
    use_saved_index = True
    sample_sets = lookup_db.retrieve_sets()
    index_sets = [IndexedSet(x, use_saved_index=use_saved_index) for x in sample_sets]
    lookup_table_1 = LookupTable(index_sets, use_saved_indexes=use_saved_index)
    return lookup_table_1

