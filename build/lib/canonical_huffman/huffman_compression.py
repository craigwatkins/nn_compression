
from .dictionary import HuffmanDictionary
from .padded_binary import write_padded_bytes, read_padded_bytes

class HuffmanCoding:
    """
    Huffman compression with canonical codes. Converts a list of integers into Huffman codes with a dictionary
    and header information.
    """
    def __init__(self):
        """
        Initializes the class
        Properties:
        huff_codes : dictionary
            symbol:huffman code
        comp_type_len : integer
            allows for 2^8 different compression types
        bit_size_lengths_len : integer
            allows for 2^8 different lengths of codes
        bit_size_symbols_len : integer
            allows for 2^8 different symbols
        num_lengths_len : integer
            allows for 2^16 different lengths of codes
        data : list of integers
            Integer values that represent the data that needs to be compressed
        encoded_list : list of strings
            A list of Huffman codes for the data
        encoded_text : String
            encoded list as single string
        header : binary string
            The following variables as a binary string
            comp_type - the type of compression used (only one type right now)
            length_bit_size - the number of bits needed to describe
                the bit lengths of the huffman codes. e.g. 3 bits can describe
                8 different lengths of huffman codes
            symbol_bit_size - the number of bits needed to describe the symbols.
                e.g. 8 bits can describe 256 different symbols
            num_lengths - the number of bit lengths included in the file
        file_info : binary string
            header information, Huffman dictionary, and encoded data
        huff_dict : HuffmanDictionary
            A class that provides a dictionary of Huffman codes

        """
        self.huff_codes = {}  # symbol:huffman code
        self.comp_type_len = 8  # allows for 2^8 different compression types
        self.bit_size_lengths_len = 8
        self.bit_size_symbols_len = 8
        self.num_lengths_len = 16  # allows for 2^16 different lengths of codes
        self.data = None
        self.encoded_list = []
        self.encoded_text = None
        self.header = None
        self.file_info = None
        self.binaries = None
        self.huff_dict = HuffmanDictionary()

    def encode_data(self):
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
        self.encoded_list = [self.huff_dict.canonical_codes[value] for value in self.data]
        self.encoded_text = ''.join(self.encoded_list)

    def build_file_header(self):
        """
        Creates header information to be used by the decoder
        -------
        header : binary string
            The following variables as a binary string
            comp_type - the type of compression used (only one type right now)
            length_bit_size - the number of bits needed to describe
                the bit lengths of the huffman codes. e.g. 3 bits can describe
                8 different lengths of huffman codes
            symbol_bit_size - the number of bits needed to describe the symbols.
                e.g. 8 bits can describe 256 different symbols
            num_lengths - the number of bit lengths included in the file
        """
        highest_value = max(self.data)
        comp_type = ("{0:0" + str(self.comp_type_len)+"b}").format(0)
        length_bit_size = ("{0:0" + str(self.bit_size_lengths_len)+"b}").format(len(self.huff_dict.compress_lengths[0]))
        symbol_bits = len("{0:b}".format(highest_value))
        symbol_bit_size = ("{0:0" + str(self.bit_size_symbols_len)+"b}").format(symbol_bits)
        num_lengths = ("{0:0" + str(self.num_lengths_len)+"b}").format(self.huff_dict.num_lengths)
        self.header = comp_type + length_bit_size + symbol_bit_size + num_lengths


    def compress(self, data, fixed_dict=False):
        """
        Parameters
        ----------
        data : list of ints
            data to be encoded
        fixed_dict : dictionary, optional
            symbol:huffman code. The default is None.

        Returns
        -------
        file_info : binary string
            header information, Huffman dictionary, and encoded data
        encoded_list : list of binary strings
            the encoded information in list format

        """
        self.data = data
        if fixed_dict is False:
            self.huff_dict.make_dictionary(self.data)
        self.encode_data()
        if fixed_dict is False:
            self.build_file_header()
            canonical_dict = ''.join(self.huff_dict.compress_lengths) + ''.join(self.huff_dict.sorted_symbols)
            self.file_info = self.header + canonical_dict + self.encoded_text

    def save_compressed(self, file_name):
        """
        Save the file_info to a binary file

        Parameters
        ----------
        file_name : string
            name and directory of file where it will be saved

        Returns
        -------
        None.

        """
        write_padded_bytes(self.file_info, file_name)
    def open_compressed(self, file_name):
        """
        Open a binary file and read the file_info

        Parameters
        ----------
        file_name : string
            name and directory of file where it will be saved

        Returns
        -------
        None.

        """
        self.binaries = read_padded_bytes(file_name)

    def decode_data(self, canonical_codes, binaries):
        """
        Use the canonical_code dictionary to get the original symbols from the
        codes. Since we don't know the size of the code, we have to add
        bits to the code one at a time until we find a match

        Parameters
        ----------
        canonical_codes : dictionary
            canonical Huffman code: symbol
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
                decoded_list.append(symbol)
                current_code = ""
        return decoded_list

    def decompress_file(self, fixed_dict=None):
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
        comp_type = int(self.binaries[:self.comp_type_len], 2)
        binaries = self.binaries[self.comp_type_len:]
        bit_size_lengths = int(binaries[:self.bit_size_lengths_len], 2)
        binaries = binaries[self.bit_size_lengths_len:]
        bit_size_symbols = int(binaries[:self.bit_size_symbols_len], 2)
        binaries = binaries[self.bit_size_symbols_len:]
        num_lengths = int(binaries[:self.num_lengths_len], 2)
        binaries = binaries[self.num_lengths_len:]
        lengths_end = bit_size_lengths*num_lengths
        lengths_bits = binaries[:lengths_end]
        binaries = binaries[lengths_end:]
        lengths = [int(lengths_bits[i:i+bit_size_lengths], 2)
                   for i in range(0, num_lengths*bit_size_lengths, bit_size_lengths)]
        num_symbols = sum(lengths)
        symbols_end = num_symbols*bit_size_symbols
        symbol_bits = binaries[:symbols_end]
        binaries = binaries[symbols_end:]
        symbols = [int(symbol_bits[i:i+bit_size_symbols], 2)
                   for i in range(0, num_symbols*bit_size_symbols, bit_size_symbols)]
        canonical_codes = self.huff_dict.decompress_canonical_dict(lengths, symbols)
        decoded_data = self.decode_data(canonical_codes, binaries)
        return decoded_data



