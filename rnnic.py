import numpy as np
import cv2 as cv
from numba import jit


from header import Header
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from canonical_huffman import HuffmanCoding as HuffmanCoding
from canonical_huffman import padded_binary as pb
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
        self.match_counter = [0] * (len(self.lookup_table.set_list) + 1)
        self.original_values = np.array([])
        self.compressed_values = np.array([])
        self.compressed_indexes = []
        self.error_threshold = 0
        self.clip_min = 0
        self.clip_max = 255
        self.matches = []
        self.height = 0
        self.width = 0
        self.row_zipper_length = 0
        self.decompressed_values = None
        self.compressed_length = 0
        self.uncompressed_bit_size = 0
        self.compressed_path = ''
        self.row_idx = 0
        self.row_matches = []

    def get_lookup_table(self):
        """
        Description: This method is used to get the lookup tables.
        """
        from create_lookup_tables import create_lookup_table

        return create_lookup_table()

    def compress(self, source_path, compressed_path, error_threshold=2):
        """
        Description: This method is used to compress the image.
        param: source_path: The path to the image to be compressed.
        param: compressed_path: The path to the binary file to save the compressed data to.
        param: error_threshold: The maximum error allowed for matches before trying a smaller pixel string size.
        Decreasing this value will usually increase quality at the expense of compression ratio.
        param: search_depth: This controls the threshold for identifying possible matches.
        A higher value usually increases the compression ratio, but also increases
        the time required to compress the image.
        """
        self.compressed_path = compressed_path
        self.error_threshold = error_threshold
        self.preprocess_image(source_path)
        self.compressed_values = np.zeros(self.original_values.shape, dtype=np.int32)
        # add an extra row at the top of compressed_values so that it can provide a basis for the first row
        self.compressed_values[0, :] = self.DEFAULT_ROW_VALUE
        # add an extra three columns at the left of compressed_values so that it can provide a basis for the first column
        self.compressed_values[:, :3] = self.DEFAULT_ROW_VALUE
        self.clip_min = min(self.original_values.flatten())
        self.clip_max = max(self.original_values.flatten())
        self.get_best_matches()
        # remove the top row of compressed_values, it's not actually part of the image
        self.compressed_values = np.delete(self.compressed_values, 0, axis=0)
        self.original_values = np.delete(self.original_values, 0, axis=0)
        # remove the left columns of compressed_values, it's not actually part of the image
        self.compressed_values = np.delete(self.compressed_values, np.s_[:3], axis=1)
        self.original_values = np.delete(self.original_values, np.s_[:3], axis=1)

        image_data = self.huff_compress()
        header_obj = Header()
        header = header_obj.build_header([self.compressed_values.shape[0], self.compressed_values.shape[1], self.row_zipper_length,
                                          self.compressed_length, self.clip_min, self.clip_max])
        # save the compressed image
        pb.write_padded_bytes(header + image_data, self.compressed_path)
        # reshape compressed values for image display
        self.compressed_values = self.compressed_values.reshape(self.compressed_values.shape[0],
                                                                self.compressed_values.shape[1] // 3, 3)

        return self.compressed_values

    def preprocess_image(self, source_path):
        """
        Description: This method is used to preprocess the image. It sets default values and calculates a
        search heuristic to find the best matches for the image.
        """
        self.original_values = cv.imread(source_path).astype(np.int16)
        self.original_values = np.insert(self.original_values, 0, self.DEFAULT_ROW_VALUE, axis=0)
        # add left columns with default values
        self.original_values = np.insert(self.original_values, 0, self.DEFAULT_ROW_VALUE, axis=1)
        self.height = self.original_values.shape[0]
        # flatten rgb channels into each row   [r1,g1,b1,r2,g2,b2,...]
        self.original_values = self.original_values.reshape(self.height, -1)
        self.width = self.original_values.shape[1]

    def get_best_matches(self):
        a_set = self.lookup_table.set_list[0]
        a_set2 = self.lookup_table.set_list[-1]
        errors = []
        for row_idx in range(1, self.height):
            for col_idx in range(3, self.width, 3):
                left = self.compressed_values[row_idx, col_idx - 3:col_idx]
                current = self.original_values[row_idx, col_idx:col_idx + 3]
                top = self.compressed_values[row_idx - 1, col_idx:col_idx + 3]
                avg = (top + left) // 2
                diffs = avg - current
                reference = avg
                index, distance = a_set.get_matches(diffs)
                match = a_set.vectors[index]
                error_threshold = self.error_threshold
                errors.append(a_set.vectors[index] - diffs)
                if distance < error_threshold:
                    self.compressed_indexes.append(self.lookup_table.index_dict[match])
                    self.compressed_values[row_idx, col_idx:col_idx + 3] = np.clip(reference - match, self.clip_min,
                                                                                   self.clip_max)
                else:
                    index, distance = a_set2.get_matches(diffs)
                    match = a_set2.vectors[index]
                    self.compressed_indexes.append(self.lookup_table.index_dict[match])
                    self.compressed_values[row_idx, col_idx:col_idx + 3] = np.clip(reference - match, self.clip_min,
                                                                                   self.clip_max)
                self.matches.append(match)

    def huff_compress(self):
        """
        Description: This method is used to compress the image using huffman coding.
        """
        indexes = self.compressed_indexes
        indexes = np.array(indexes)
        indexes_2d = indexes.reshape(-1, self.compressed_values.shape[1] // 3)
        row_means = np.mean(indexes_2d, axis=1)
        overall_mean = np.mean(row_means)
        # get the rows that are less than the mean
        small_rows = indexes_2d[row_means < overall_mean]
        # get the indexes for the rows in indexes_2d that are less than the mean
        is_small = (row_means >= overall_mean).astype(int)
        # convert is_small to binary string
        is_small_string = ''.join([str(x) for x in is_small])
        self.row_zipper_length = len(is_small_string)
        # get the rows that are greater or equal to the mean
        large_rows = indexes_2d[row_means >= overall_mean]
        small_rows = small_rows.flatten()
        large_rows = large_rows.flatten()
        huff_small = HuffmanCoding()
        huff_large = HuffmanCoding()
        huff_small.compress(small_rows)
        huff_large.compress(large_rows)
        huff_small_compressed = huff_small.file_info
        huff_large_compressed = huff_large.file_info
        file_info = is_small_string + huff_small_compressed + huff_large_compressed
        self.compressed_length = len(huff_small_compressed)
        return file_info

    def decompress(self, file_name=''):
        """
        Description: This method is used to decompress the image.
        :param file_name: The path to the binary file to be decompressed.
        :return: None
        """
        decompressed_matches, decompressed_indexes = self.decompress_file(file_name, self.lookup_table)
        # reconstruct the image
        self.rebuild_image(decompressed_matches)

    def decompress_file(self, file_name, lookup_table):
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
        self.clip_min = header_values['clip_min']
        self.clip_max = header_values['clip_max']
        self.row_zipper_length = header_values['row_zipper_length']
        smalls_bit_length = header_values['length']
        row_zipper = full_file[header_length:header_length + self.row_zipper_length]
        current_index = header_length + self.row_zipper_length
        small_rows = full_file[current_index: current_index + smalls_bit_length]
        current_index += smalls_bit_length
        huff = HuffmanCoding()
        huff.binaries = small_rows
        decompressed_indexes_small = huff.decompress_file()
        large_rows = full_file[current_index:]
        huff = HuffmanCoding()
        huff.binaries = large_rows
        decompressed_indexes_large = huff.decompress_file(large_rows)
        index_width = self.width // 3
        decompressed_indexes = np.zeros(self.height * index_width, dtype=np.int32)
        row_zipper = list(row_zipper)
        sm_idx = 0
        lg_idx = 0
        for i, choice in enumerate(row_zipper):
            if choice == '0':
                decompressed_indexes[i * index_width:(i + 1) * index_width] = decompressed_indexes_small[sm_idx * index_width:(sm_idx + 1) * index_width]
                sm_idx += 1
            else:
                decompressed_indexes[i * index_width:(i + 1) * index_width] = decompressed_indexes_large[lg_idx * index_width:(lg_idx + 1) * index_width]
                lg_idx += 1

        # convert to vectors
        decompressed_vectors = [lookup_table.index_dict_reverse[x] for x in decompressed_indexes]

        return decompressed_vectors, decompressed_indexes

    def rebuild_image(self, matches):
        """
        Description: This method is used to rebuild the image.

        :param matches: The list of matches that will be used to rebuild the image.
        """
        decompressed_values = np.zeros((self.height, self.width+3), dtype=np.int32)
        decompressed_values = np.insert(decompressed_values, 0, self.DEFAULT_ROW_VALUE, axis=0)
        decompressed_values[:, :3] = self.DEFAULT_ROW_VALUE
        match_idx = 0
        for row_idx in range(1, decompressed_values.shape[0]):
            # get the matches for the row
            for col_idx in range(3, decompressed_values.shape[1], 3):
                match = matches[match_idx]
                above = decompressed_values[row_idx - 1, col_idx:col_idx + 3]
                left = decompressed_values[row_idx, col_idx - 3:col_idx]
                average = (np.array(left) + np.array(above)) // 2
                entry = np.clip(average - np.array(match), self.clip_min, self.clip_max)
                decompressed_values[row_idx, col_idx:col_idx + 3] = entry
                match_idx += 1

        # delete the first row of decompressed_values
        decompressed_values = np.delete(decompressed_values, 0, axis=0)
        # delete the first three columns of decompressed_values
        decompressed_values = np.delete(decompressed_values, np.s_[:3], axis=1)
        # reshape to 3 channels for image display
        self.decompressed_values = decompressed_values.reshape(decompressed_values.shape[0],
                                                               decompressed_values.shape[1] // 3, 3)


@jit(nopython=True)
def check_if_filled(mask_1, indexes, block_size):
    valid_indexes = []
    for i, idx in enumerate(indexes):
        # Ensure that none of the indexes in the block are already in the mask
        if not np.any(mask_1[idx:idx + block_size]):
            valid_indexes.append(idx)
    return valid_indexes


@jit(nopython=True)
def get_diffs(row, indexes, block_size):
    diffs = np.zeros((len(indexes), block_size))
    for i, idx in enumerate(indexes):
        diffs[i] = row[idx:idx + block_size]
    return diffs


@jit(nopython=True)
def add_to_mask(mask, indexes, block_size):
    for idx in indexes:
        mask[idx:idx + block_size] = True
    return mask


@jit(nopython=True)
def add_matches(row, matches, indexes, block_size):
    # indexes are the column indexes of the matches
    # matches are the vector matches for the block_size
    for i, idx in enumerate(indexes):
        row[idx:idx + block_size] = matches[i]
    return row


#@jit(nopython=True)
def get_non_overlapping(col_indexes, block_size):
    cur_cutoff = 0
    non_overlapping_indexes = []
    adj = block_size - 1
    for k, idx in enumerate(col_indexes):
        if idx < cur_cutoff:
            pass
        else:
            non_overlapping_indexes.append(k)
            cur_cutoff = idx + adj
    return non_overlapping_indexes
