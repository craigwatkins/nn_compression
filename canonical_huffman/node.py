class Node:
    """
    Tree node for huffman coding
    """
    def __init__(self, symbol, occurrences):
        self.symbol = symbol
        self.occurrences = occurrences
        self.l_child = None
        self.r_child = None

    def __lt__(self, other):
        return self.occurrences < other.occurrences