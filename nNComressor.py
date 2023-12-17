import numpy as np
import cv2 as cv

from huffman_compression import HuffmanCoding
import padded_binary as pb
from header import Header


class NNCompressor:
    """
    Description: This class is used to compress and decompress a given image.
    """

    DEFAULT_ROW_VALUE = 128

    def __init__(self):
        """
        Description: This method is used to initialize the class variables.
        """
        self.lookup_table = self.get_lookup_table()
        self.match_counter = [0] * len(self.lookup_table.set_list)
        self.original_values = np.array([])
        self.compressed_values = np.array([])
        self.error_threshold = 0
        self.clip_min = 0
        self.clip_max = 255
        self.matches = []
        self.height = 0
        self.width = 0
        self.match_bit_size = 8
        self.decompressed_values = None
        self.compressed_bit_length = 0
        self.uncompressed_bit_length = 0
        self.compressed_path = ''

    def get_lookup_table(self):
        """
        Description: This method is used to get the lookup tables.
        """
        from create_lookup_tables import create_lookup_table
        return create_lookup_table()

    def compress(self, source_path, compressed_path, error_threshold=2):
        """
        Description: This method is used to compress the image.
        """
        self.compressed_path = compressed_path
        self.error_threshold = error_threshold
        self.preprocess_image(source_path)
        self.compressed_values = np.zeros(self.original_values.shape, dtype=np.int32)
        # add an extra row at the top of compressed_values so that it can provide a basis for the first row
        self.compressed_values[0, :] = self.DEFAULT_ROW_VALUE
        self.clip_min = min(self.original_values.flatten())
        self.clip_max = max(self.original_values.flatten())
        self.height = self.original_values.shape[0]
        self.width = self.original_values.shape[1]

        self.get_best_matches()
        # remove the top row of compressed_values, it's not actually part of the image
        self.compressed_values = np.delete(self.compressed_values, 0, axis=0)
        image_data = self.compress_first_bits(self.matches, self.match_bit_size)
        header_obj = Header()
        header = header_obj.build_header([self.compressed_values.shape[0], self.compressed_values.shape[1],
                                          self.compressed_bit_length, self.uncompressed_bit_length,
                                          self.clip_min, self.clip_max])
        # save the compressed image
        pb.write_padded_bytes(header+image_data, self.compressed_path)
        print("match counter:", self.match_counter)
        # reshape compressed values for image display
        self.compressed_values = self.compressed_values.reshape(self.compressed_values.shape[0], self.compressed_values.shape[1]//3, 3)
        return self.compressed_values

    def preprocess_image(self, source_path):
        img = cv.imread(source_path)
        # flatten rgb channels into each row   [r1,g1,b1,r2,g2,b2,...]
        combined_channels = img.reshape(img.shape[0], -1)
        # add an extra row at the top of combined_channels
        # this aligns it with the rows in the compressed_values array
        combined_channels = np.insert(combined_channels, 0, self.DEFAULT_ROW_VALUE, axis=0)
        # convert original values to avoid overflow errors
        self.original_values = combined_channels.astype(np.int32)

    def get_best_matches(self):
        """
        Description: This method is used to iterate over the image and find the difference between the actual values of
        the current row and the approximated values of the previous row. The differences are approximated by finding the closest match in the lookup table.
        The closest match is then used to generate the new values for the current row which is then stored in the
        compressed_values array.
        """
        max_block_size = self.lookup_table.max_block_size
        for row_idx in range(1, self.height):
            col_idx = 0
            row_diffs = self.compressed_values[row_idx - 1, :] - self.original_values[row_idx, :]
            while col_idx < self.width:
                if row_idx == 1 and col_idx > 0:
                    # generate values for the first (extra) row of compressed_values so that actual values can be
                    # subtracted from them to get the differences for the first (actual) row of the image
                    self.approximate_top_values(row_idx, col_idx, self.compressed_values)
                    row_diffs[col_idx:col_idx + max_block_size] = self.compressed_values[row_idx - 1, col_idx:col_idx + max_block_size] - self.original_values[row_idx, col_idx:col_idx + max_block_size]
                match, size = self.get_match(col_idx, row_diffs)
                self.matches.append(match)
                neighbors = np.array(self.compressed_values[row_idx - 1, col_idx:col_idx + size])
                self.compressed_values[row_idx, col_idx:col_idx + size] = np.clip(neighbors - np.array(match), self.clip_min, self.clip_max)
                col_idx += size

    def get_match(self, col_idx, row_diffs):
        block_size = 0
        closest_match = None
        i = 0
        for i, a_set in enumerate(self.lookup_table.set_list):
            block_size = a_set.block_size
            if col_idx + block_size > self.width:
                # if the there isn't enough room for a vector of this size to fit in the row
                # move on to the next smaller block size
                continue
            diff = row_diffs[col_idx:col_idx+block_size]
            diff_size = np.linalg.norm(diff)
            if diff_size > self.error_threshold*block_size and block_size > 9:
                # if the difference is too large, move on to the next smaller vector size
                # the odds of finding a good match are low and this will save time
                continue
            closest_match = tuple(a_set.set_index.get_closest_match(diff))
            error = np.linalg.norm(diff - np.array(closest_match))
            under_threshold = error < self.error_threshold
            if under_threshold:
                self.match_counter[i] += 1
                return closest_match, block_size
            else:
                continue
        self.match_counter[i] += 1
        return closest_match, block_size

    def compress_first_bits(self, values, bit_size):
        """
        Description: compress the first bit_size bits of values using Huffman compression
        and combine them with the remaining bits into a single binary string
        """
        indexes = [self.lookup_table.index_dict[tuple(x)] for x in values]
        indexes = np.array(indexes)
        max_index = self.lookup_table.max_index
        num_bits = len(bin(max_index)[2:])
        self.uncompressed_bit_length = num_bits - bit_size
        # convert the indexes to bits and take the first bit_size bits of each index
        # (the first bit_size bits are the most significant bits)
        # pad the bits with 0s if necessary to make them all the same length (num_bits)
        diffs_bits = [bin(x)[2:].zfill(num_bits)[:bit_size] for x in indexes]
        # get the remaining bits (the least significant bits) and add them to a new list
        remaining_bits = [bin(x)[2:].zfill(num_bits)[bit_size:] for x in indexes]
        # convert the remaining bits to a single binary string
        remaining_bits = ''.join(remaining_bits)
        # The huffman class expects decimal values, so convert the bits to decimal
        diffs_decimals = [int(x, 2) for x in diffs_bits]
        huff = HuffmanCoding()
        huff_compressed, encoded_list = huff.compress(diffs_decimals)
        self.compressed_bit_length = len(huff_compressed)
        file_info = huff_compressed + remaining_bits
        return file_info

    def approximate_top_values(self, row_idx, col_idx, destination_values):
        """
        Description: This method is used to create new values that close to the actual values of the current row.
        By using the previous values in the row. Used for both compression and decompression.
        :param row_idx:
        :param col_idx:
        :param destination_values:  the array of values that need to be approximated.
        :return:
        """
        max_block_size = self.lookup_table.max_block_size
        if self.width - col_idx < max_block_size:
            max_block_size = self.width - col_idx
        values = destination_values[row_idx, col_idx - 3:col_idx].tolist()  # get the previous three values
        # fill neighbors with values from previous row, values is a list of three values that are repeated to fill
        approximate_neighbors = [values[j] for i in range(max_block_size//3) for j in range(3)]
        destination_values[row_idx - 1, col_idx:col_idx + max_block_size] = approximate_neighbors

    def decompress(self, file_name=''):
        decompressed_matches, decompressed_indexes = self.decompress_file(file_name, self.match_bit_size, self.lookup_table)
        # reconstruct the image
        self.rebuild_image(decompressed_matches)

    def decompress_file(self, file_name, bit_size, lookup_table):
        """
        Description: decompress the first bit_size bits of values
        """
        full_file = pb.read_padded_bytes(file_name)
        header_obj = Header()
        header_length = header_obj.get_total_length()
        header = full_file[:header_length]
        header_values = header_obj.decompress_header_values(header)
        self.height = header_values['height']
        self.width = header_values['width']
        compressed_bit_length = header_values['length']
        uncompressed_bit_size = header_values['uncompressed_bit_size']
        first_bits = full_file[header_length:header_length+compressed_bit_length]
        uncompressed_bits = full_file[header_length+compressed_bit_length:]
        huff = HuffmanCoding()
        first_bits = huff.decompress_file(first_bits)
        if not first_bits or not uncompressed_bits:
            print(f"""Error: Could not open file""")
            return None, None
        # convert the integers to bits and pad them with 0s if necessary to make them all the same length (bit_size)
        decompressed_first_bits = [bin(x)[2:].zfill(bit_size) for x in first_bits]
        # break the remaining bits into groups of remaining_match_bit_size bits
        uncompressed_bits = [uncompressed_bits[i:i + uncompressed_bit_size] for i in range(0, len(uncompressed_bits), uncompressed_bit_size)]
        # combine decompressed_first_bits and remaining_matches_bits into a single list
        decompressed = [decompressed_first_bits[i] + uncompressed_bits[i] for i in range(len(decompressed_first_bits))]
        # convert to decimal
        decompressed_indexes = [int(x, 2) for x in decompressed]
        # convert to vectors
        decompressed_vectors = [lookup_table.index_dict_reverse[x] for x in decompressed_indexes]

        return decompressed_vectors, decompressed_indexes

    def rebuild_image(self, matches):
        """
        Description: This method is used to rebuild the image.
        """
        decompressed_values = np.zeros((self.height, self.width), dtype=np.int32)
        decompressed_values = np.insert(decompressed_values, 0, 128, axis=0)
        match_idx = 0
        for row_idx in range(1, decompressed_values.shape[0]):
            # get the matches for the row
            col_idx = 0
            while col_idx < self.width:
                if row_idx == 1 and col_idx > 0:
                    self.approximate_top_values(row_idx, col_idx, decompressed_values)
                match = matches[match_idx]
                block_size = len(match)
                decompressed_values[row_idx, col_idx:col_idx+block_size] = np.clip(decompressed_values[row_idx-1, col_idx:col_idx+block_size] - np.array(match), self.clip_min, self.clip_max)
                col_idx += block_size
                match_idx += 1
        # delete the first row of decompressed_values
        decompressed_values = np.delete(decompressed_values, 0, axis=0)
        # reshape to 3 channels for image display
        self.decompressed_values = decompressed_values.reshape(decompressed_values.shape[0], decompressed_values.shape[1]//3, 3)
