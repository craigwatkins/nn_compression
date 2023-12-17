"""
Huffman compression with canonical codes

"""
import heapq
from collections import Counter
import bisect


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


class HuffmanCoding:
    """
    Provides Huffman coding for data

    """

    def __init__(self):

        self.huff_codes = {}  # symbol:huffman code
        self.comp_type_len = 8
        self.bit_size_lengths_len = 8
        self.bit_size_symbols_len = 8
        self.num_lengths_len = 16

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
            # add lowest nodes as children
            summed.l_child = node_a
            summed.r_child = node_b
            # add the summed node to the heap
            heapq.heappush(node_heap, summed)
        return node_heap

    def make_code_dict(self, node, current_code):
        """
        Builds a dictionary in self.huff_codes (symbol:code) by traversing the tree

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

    def get_encoded_data(self, data):
        """
        Convert the data into Huffman codes

        Parameters
        ----------
        data : list of ints
            Integer values that represent the data that needs to be compressed

        Returns
        -------
        encoded_text : String
            encoded list as single string
        encoded_list : List of strings
            A list of Huffman codes for the data
        """
        encoded_list = [self.huff_codes[value] for value in data]
        encoded_text = ''.join(encoded_list)
        return encoded_text, encoded_list

    def make_canonical_codes(self):
        """
        Convert the generated Huffman codes to canonical codes and return
        the dictionary information necessary for decompression.

        Returns
        -------
        canonical_codes : Dictionary
            symbol:canonical code
        sorted_symbols : Two dimensional list of binary strings
            Each sublist grouped by bit size, then sorted in lexigraphical order
        compress_lengths : List of integers
            The number of bits for each sublist of sorted symbols, zero values
            are used if there are no symbols of a specific length
        num_lengths : integer
            The size of the highest length. This will be needed to create
            compress_lengths

        """
        symbols_by_length = {}  # list of lists of codes, each sublist by length
        canonical_codes = {}  # symbol:canonical code
        sorted_symbols = []  # list of codes,by bit size, then lexigraphical order
        # create a list of lists, each list containing sorted symbols
        # with the same bit length
        for symbol in self.huff_codes:
            length = len(self.huff_codes[symbol])
            if length in symbols_by_length:
                symbol_list = symbols_by_length[length]
                bisect.insort(symbol_list, symbol)
            else:
                symbols_by_length[length] = [symbol]
        sorted_lengths = sorted(symbols_by_length.keys())
        max_len = max(sorted_lengths)
        compress_lengths = [len(
            symbols_by_length[i]) if i in symbols_by_length else 0 for i in range(0, max_len + 1)]
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
                canonical_codes[symbols_by_length[length][i]] = canonical_code
                sorted_symbols.append(symbols_by_length[length][i])
            prev_code = prev_code + i
        num_lengths = len(compress_lengths)
        compress_lengths = self.make_smallest_binary(compress_lengths)
        sorted_symbols = self.make_smallest_binary(sorted_symbols)
        return canonical_codes, sorted_symbols, compress_lengths, num_lengths

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
        bit_diff = 0
        symbol_index = 0
        first_code = False
        for i, num_symbols in enumerate(compress_lengths):
            # i: number of bits in symbol
            format_string = "{0:0" + str(i) + "b}"
            if num_symbols > 0:
                if i > 0 and first_code is True:
                    prev_code = prev_code + 1 << bit_diff
                else:
                    prev_code = prev_code << bit_diff
                    first_code = True
                for j in range(num_symbols):
                    canonical_code = format_string.format(prev_code + j)
                    # canonical_code = prev_code + j
                    canonical_codes[canonical_code] = sorted_symbols[symbol_index]
                    symbol_index += 1
                prev_code = prev_code + j
                bit_diff = 0
            bit_diff += 1
        return canonical_codes

    def get_bit_size(self, num):
        # returns the bit size needed to represent the integer
        return len("{0:b}".format(num))

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

    def build_file_header(self, data, compress_lengths, num_lengths):
        """
        Creates header information to be used by the decoder

        Parameters
        ----------
        data : List of integers
            data to be encoded
        compress_lengths : List of integers
            The number of bits for each sublist of sorted symbols, zero values
            are used if there are no symbols of a specific length
        num_lengths : integer
            The size of the highest length. This will be needed to create
            compress_lengths

        Returns
        -------
        header : binary string
            binary string of the following variables
            comp_type - the type of compression used (only one type right now)
            length_bit_size - the number of bits needed to describe
                the bit lengths of the huffman codes
            symbol_bit_size - the number of bits needed to describe the symbols
            num_lengths - the number of bit lengths included in the file
            eof - the end of file value that is based on the max data value + 1
        """
        highest_value = max(data)
        comp_type = ("{0:0" + str(self.comp_type_len) + "b}").format(0)
        length_bit_size = ("{0:0" + str(self.bit_size_lengths_len) + "b}").format(len(compress_lengths[0]))
        symbol_bits = len("{0:b}".format(highest_value))
        symbol_bit_size = ("{0:0" + str(self.bit_size_symbols_len) + "b}").format(symbol_bits)
        num_lengths = ("{0:0" + str(self.num_lengths_len) + "b}").format(num_lengths)
        format_string = "{0:0" + str(symbol_bits + 1) + "b}"
        eof = format_string.format(highest_value + 1)
        header = comp_type + length_bit_size + symbol_bit_size + num_lengths + eof
        return header

    def compress(self, data, fixed_dict=None):
        """
        Parameters
        ----------
        data : list of ints
            data to be encoded
        fixed_dict : dictionary, optional
            symbol:huffman code The default is None.

        Returns
        -------
        file_info : binary string
            header information, Huffman dictionary, and encoded data
        encoded_list : list of binary strings
            the encoded information in list format

        """
        #print("Compressing into Huffman code...")
        fixed = False
        if fixed_dict is None:
            # get the all of the occurrences of each symbol
            occurrences = Counter(data)
            # make nodes with the dictionary and add them to the heap
            node_heap = self.make_heap(occurrences)
        else:
            fixed = True
            node_heap = self.make_heap(fixed_dict)

        node_heap = self.build_tree(node_heap)
        root = heapq.heappop(node_heap)
        self.make_code_dict(root, "")  # makes "regular" Huffman codes
        canonical_codes, sorted_symbols, compress_lengths, num_lengths = self.make_canonical_codes()
        self.huff_codes = canonical_codes
        encoded_text, encoded_list = self.get_encoded_data(data)

        if fixed is False:
            header = self.build_file_header(data, compress_lengths, num_lengths)
            canonical_dict = ''.join(compress_lengths) + ''.join(sorted_symbols)
            file_info = header + canonical_dict + encoded_text
            print('dictionary size: ', len(canonical_dict) / 8, ' bytes')
            #print("total compressed size: ",
                  #(len(header) + len(encoded_text) + len(canonical_dict)) / 8, 'bytes')
        else:
            file_info = encoded_text
            #print("compressed data size: ", len(encoded_text) / 8, 'bytes')
        return file_info, encoded_list

    def decode_data(self, canonical_codes, eof, binaries):
        """
        Use the canonical_code dictionary to get the original symbols from the
        codes. Since we don't know the size of the code, we have to add
        bits to the code one at a time until we find a match

        Parameters
        ----------
        canonical_codes : dictionary
            canonical Huffman code: symbol
        eof : binary string
            The code used to indicate that we have reached the end of the file
        binaries : binary string
            data encoded with Huffman codes

        Returns
        -------
        decoded_list : list of integers
            the integers that were originally encoded

        """
        current_code = ""
        decoded_list = []
        for bit in binaries:
            current_code += bit
            if current_code in canonical_codes:
                symbol = canonical_codes[current_code]
                if symbol != eof:
                    decoded_list.append(symbol)
                else:
                    break
                current_code = ""
        return decoded_list

    def decompress_file(self, binaries, fixed_dict=None):
        """
        TO DO: add different compression options for Huffman dictionaries

        Parameters
        ----------
        binaries : binary string
            data from file
        fixed_dict : dictionary, optional
            A dictionary of Huffman codes from an external source.
            The default is None.

        Returns
        -------
        decoded_data : list of ints
            The original data that was encoded

        """

        comp_type = int(binaries[:self.comp_type_len], 2)
        if comp_type == 0 and fixed_dict is None:
            binaries = binaries[self.comp_type_len:]
            bit_size_lengths = int(binaries[:self.bit_size_lengths_len], 2)
            binaries = binaries[self.bit_size_lengths_len:]
            bit_size_symbols = int(binaries[:self.bit_size_symbols_len], 2)
            binaries = binaries[self.bit_size_symbols_len:]
            num_lengths = int(binaries[:self.num_lengths_len], 2)
            binaries = binaries[self.num_lengths_len:]
            eof_len = bit_size_symbols + 1
            eof = binaries[:eof_len]
            binaries = binaries[eof_len:]
            lengths_end = bit_size_lengths * num_lengths
            lengths_bits = binaries[:lengths_end]
            binaries = binaries[lengths_end:]
            lengths = [int(lengths_bits[i:i + bit_size_lengths], 2)
                       for i in range(0, num_lengths * bit_size_lengths, bit_size_lengths)]
            num_symbols = sum(lengths)
            symbols_end = num_symbols * bit_size_symbols
            symbol_bits = binaries[:symbols_end]
            binaries = binaries[symbols_end:]
            symbols = [int(symbol_bits[i:i + bit_size_symbols], 2)
                       for i in range(0, num_symbols * bit_size_symbols, bit_size_symbols)]
            canonical_codes = self.decompress_canonical_dict(lengths, symbols)
            decoded_data = self.decode_data(canonical_codes, eof, binaries)
            return decoded_data

        return None