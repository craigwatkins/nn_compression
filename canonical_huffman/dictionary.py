"""
Dictionary class
properties: huffman_tree, huffman_dictionary, canonical huffman dictionary
methods: build_huffman_tree, build_huffman_dictionary, build_canonical_huffman_dictionary, get_canonical_huffman_dictionary
"""
from .node import Node
import heapq
import bisect
from collections import Counter


class HuffmanDictionary:
    def __init__(self):
        """
        canonical_codes : Dictionary
            symbol:canonical code
        sorted_symbols : Two dimensional list of binary strings
            Each sublist grouped by bit size, then sorted in lexicographical order
        compress_lengths : List of integers
            Each integer represents the number of symbols of the length designated by its index
            in the list plus one (lists are zero indexed, but symbols need at least one bit)
        num_lengths : integer
            The number of different bit lengths in the canonical codes.

        """
        self.occurrences = None
        self.huffman_tree = None
        self.huffman_dictionary = None
        self.canonical_huffman_dictionary = None
        self.node_heap = None
        self.huff_codes = {}
        self.canonical_codes = {}
        self.sorted_symbols = []
        self.compress_lengths = None
        self.num_lengths = None

    def make_dictionary(self, data=None, occurrences=None):
        if data is None and occurrences is None:
            raise ValueError("No data or occurrences provided.")
        if occurrences is not None:
            self.occurrences = occurrences
        else:
            self.occurrences = Counter(data)
        self.node_heap = self.make_heap(self.occurrences)
        self.node_heap = self.build_tree(self.node_heap)
        root = heapq.heappop(self.node_heap)
        self.make_code_dict(root, "")
        self.make_canonical_codes()

    def fixed_dictionary(self, canonical_codes):
        """
        Parameters
        ----------
        canonical_codes : Dictionary
            symbol:canonical code

        Returns
        -------
        None.

        """
        self.canonical_codes = canonical_codes
        self.sorted_symbols = list(canonical_codes.keys())
        self.compress_lengths = [len(canonical_codes[x]) for x in canonical_codes]
        self.num_lengths = len(self.compress_lengths)

    def make_heap(self, occurrences):
        """
        Parameters
        ----------
        occurrences : Dictionary
            Key - Symbols represented by integer values
            Value - number of times that the symbol occurs in the data

        Returns
        -------
        None.

        """
        node_heap = []
        for key in occurrences:
            node = Node(key, occurrences[key])
            heapq.heappush(node_heap, node)
        return node_heap

    def build_tree(self, node_heap):
        """
        Builds the Huffman tree with node_heap
        final result is a list with only the root node

        Returns
        -------
        None.

        """
        while len(node_heap) > 1:
            # pop the two lowest nodes from the heap
            node_a = heapq.heappop(node_heap)
            node_b = heapq.heappop(node_heap)
            # create a new node from their sum
            summed = Node(None, node_a.occurrences + node_b.occurrences)
            # add the lowest nodes as children
            summed.l_child = node_a
            summed.r_child = node_b
            # add the summed node to the heap
            heapq.heappush(node_heap, summed)
        return node_heap

    def make_code_dict(self, node, current_code):
        """
        Makes "regular" Huffman codes. Builds a dictionary (symbol:code) by traversing the tree

        Parameters
        ----------
        node : instance of Node class
            nodes contains the children and occurrences for each symbol in data
        current_code : binary string
            current code for this place in the binary tree

        Returns
        -------
        None.

        """
        if node is None:
            return
        if node.symbol is not None:
            self.huff_codes[node.symbol] = current_code
            return
        self.make_code_dict(node.l_child, current_code + "0")
        self.make_code_dict(node.r_child, current_code + "1")

    def make_canonical_codes(self):
        """
        Convert the generated Huffman codes to canonical codes and return
        the dictionary information necessary for decompression.
        """
        symbols_by_length = {}  # list of lists of codes, each sublist by length
        # create a dictionary of lists, each list containing sorted symbols with the same bit length
        # with the same bit length
        for symbol in self.huff_codes:
            length = len(self.huff_codes[symbol])
            if length in symbols_by_length:
                symbol_list = symbols_by_length[length]  # get the list of symbols with the same length
                bisect.insort(symbol_list, symbol)  # insert symbol in sorted order into the appropriate list
            else:
                symbols_by_length[length] = [symbol]
        sorted_lengths = sorted(symbols_by_length.keys())  # get the sorted list of bit lengths
        max_len = max(sorted_lengths)
        # get the number of symbols for each bit length and store in a list
        # each index represents the bit length and the value is the number of symbols of that length
        compress_lengths = [len(symbols_by_length[i]) if i in symbols_by_length else 0 for i in range(1, max_len + 1)]
        prev_code = 0
        # increment each successive code by one
        # add another one and leftshift when a new code length is reached
        for j, length in enumerate(sorted_lengths):
            format_string = "{0:0" + str(length) + "b}"
            if j > 0:
                prev_code = prev_code + 1 << (length - sorted_lengths[j - 1])
            num_codes = len(symbols_by_length[length])
            for i in range(num_codes):
                canonical_code = format_string.format(prev_code + i)
                self.canonical_codes[symbols_by_length[length][i]] = canonical_code
                self.sorted_symbols.append(symbols_by_length[length][i])
            prev_code = prev_code + i
        self.num_lengths = len(compress_lengths)
        self.compress_lengths = self.make_smallest_binary(compress_lengths)
        self.sorted_symbols = self.make_smallest_binary(self.sorted_symbols)

    def make_smallest_binary(self, ints):
        """
        takes a list of integers and converts them to a list of binary numbers
        of the minimum required bit size for the largest and then converts
        those numbers into equally sized binary strings

        Parameters
        ----------
        ints : List of ints
            DESCRIPTION.

        Returns
        -------
        binary_list : list of binary strings

        """
        max_int = max(ints)
        binary_list = []
        # get size of binary conversion
        compressed_bit_length = len("{0:b}".format(max_int))
        format_string = "{0:0" + str(compressed_bit_length) + "b}"
        for num in ints:
            # convert index to proper sized binary
            binary = format_string.format(num)
            binary_list.append(binary)
        return binary_list

    def decompress_canonical_dict(self, compress_lengths, sorted_symbols):
        """
        Parameters
        ----------
        compress_lengths : List of integers
            The number of bits for each sublist of sorted symbols, zero values
            are used if there are no symbols of a specific length
        sorted_symbols : Two dimensional list of binary strings
            Each sublist grouped by bit size, then sorted in lexigraphical order

        Returns
        -------
        canonical_codes : Dictionary
             canonical code: symbol, reverse of compression dict
        """

        canonical_codes = {}
        prev_code = 0
        bit_diff = 0  # difference in bit length between successive codes
        symbol_index = 0
        first_code_found = False

        for i in range(0, len(compress_lengths)):
            num_symbols = compress_lengths[i]
            num_bits_in_code = i + 1
            format_string = "{0:0"+str(num_bits_in_code)+"b}"
            # if there are symbols of this length then we need to add the next code
            if num_symbols > 0:
                # if this is not the first code of this length then we need to left shift the previous code
                if i > 0 and first_code_found is True:
                    # left shift the previous code plus 1 by the difference in bit length
                    prev_code = prev_code + 1 << bit_diff
                else:  # if this is the first code of this length then we need to left shift the previous code
                    prev_code = prev_code << bit_diff
                    first_code_found = True
                # add the codes of this length to the dictionary
                for j in range(num_symbols):
                    canonical_code = format_string.format(prev_code + j)
                    canonical_codes[canonical_code] = sorted_symbols[symbol_index]
                    symbol_index += 1
                prev_code = prev_code + j
                bit_diff = 0
            bit_diff += 1
        return canonical_codes

"""
dict1 = HuffmanDictionary()
dict1.make_dictionary([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dict1.canonical_codes)
compress_lengths = [int(x, 2) for x in dict1.compress_lengths]
sorted_symbols = [int(x, 2) for x in dict1.sorted_symbols]
dict2 = HuffmanDictionary()
codes = dict2.decompress_canonical_dict(compress_lengths, sorted_symbols)
print(codes)
"""