import numpy as np
import cv2 as cv
import time
from scipy.spatial import KDTree
from numba import jit


from huffman_compression import HuffmanCoding
import padded_binary as pb
from header import Header


class NNCompressor:
    """
    Description: This class is used to compress and decompress a given image.
    """

    DEFAULT_ROW_VALUE = 3

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
        self.decompressed_values = None
        self.max_compressed_bit_size = 14
        self.compressed_length = 0
        self.uncompressed_bit_size = 0
        self.compressed_path = ''
        self.row_idx = 0
        self.specials = [[0], [0], [0]]
        self.one_pixel_set = self.lookup_table.set_list[-1].vectors
        self.kd_tree = KDTree(self.one_pixel_set)
        self.row_matches = []
        self.block_index_sets = []
        self.trees = [KDTree(a_set.vectors) for a_set in self.lookup_table.set_list]
        self.row_lookup_indexes = {}

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
        start = time.time()
        self.preprocess_image(source_path)
        end = time.time()
        print("time to preprocess", end - start)
        self.compressed_values = np.zeros(self.original_values.shape, dtype=np.int32)
        # add an extra row at the top of compressed_values so that it can provide a basis for the first row
        self.compressed_values[0, :] = self.DEFAULT_ROW_VALUE
        self.clip_min = min(self.original_values.flatten())
        self.clip_max = max(self.original_values.flatten())

        start = time.time()
        self.get_best_matches()
        end = time.time()
        print("time to get best matches", end - start)
        print("specials:", self.specials)
        # remove the top row of compressed_values, it's not actually part of the image
        self.compressed_values = np.delete(self.compressed_values, 0, axis=0)

        image_data = self.huff_compress()
        header_obj = Header()
        header = header_obj.build_header([self.compressed_values.shape[0], self.compressed_values.shape[1],
                                          self.compressed_length, self.uncompressed_bit_size,
                                          self.clip_min, self.clip_max])
        # save the compressed image
        pb.write_padded_bytes(header + image_data, self.compressed_path)
        print("match counter:", self.match_counter)

        # reshape compressed values for image display
        self.compressed_values = self.compressed_values.reshape(self.compressed_values.shape[0],
                                                                self.compressed_values.shape[1] // 3, 3)

        return self.compressed_values

    def preprocess_image(self, source_path):
        self.original_values = cv.imread(source_path).astype(np.int16)
        # insert a row of default values at the top of the image
        self.original_values = np.insert(self.original_values, 0, self.DEFAULT_ROW_VALUE, axis=0)
        self.height = self.original_values.shape[0]
        self.width = self.original_values.shape[1]
        top_diffs = self.original_values[1:] - self.original_values[:-1]
        top_diffs_flat = top_diffs.reshape(top_diffs.shape[0]*top_diffs.shape[1], -1)
        # find the Euclidean distance of each r,g,b vector
        top_diff_sizes = np.linalg.norm(top_diffs_flat, axis=1)
        # reshape to a list of r,g,b vectors
        #top_diffs = top_diffs.reshape(top_diffs.shape[0]*top_diffs.shape[1], -1)
        threshold_mults = [3, 3, 3, 3, 1000, 1000, 1000, 4]
        matches = [0]*(len(self.lookup_table.set_list) - 1)
        """
        Try to find runs of small diffs appropriate for each block size across the entire image
        The heuristic is to sum the Euclidean distances of the r, g, b diffs in a block, and if the sum 
        is below a threshold, then the block is a candidate for matching.
        Record the starting indexes of the blocks that are below the threshold by block size and by row
        """
        for i in range(len(self.lookup_table.set_list)-1):
            indices_by_row = [[] for _ in range(self.height - 1)]
            a_set = self.lookup_table.set_list[i]
            block_size = a_set.block_size//3
            threshold = threshold_mults[i] * self.error_threshold
            # use a convolution to find the sums of the diffs in range block_size
            kernel = np.ones(block_size, dtype=int)
            padding_needed = len(kernel) - 1
            top_diff_sizes_convolved = np.convolve(top_diff_sizes, kernel, 'valid')
            top_diff_sizes_convolved = np.pad(top_diff_sizes_convolved, (0, padding_needed), mode='constant', constant_values=99999).reshape(self.height-1, self.width)
            # find the starting indices of blocks where the convolution result is below threshold
            # block starts has two arrays, the first is the row indexes, the second is the column indexes
            block_starts = np.where(top_diff_sizes_convolved < threshold)
            matches[i] = len(block_starts[0])
            for j, row in enumerate(block_starts[0]):  # iterate through the row indexes
                # the end of a row does not have valid values from the kernel, and cannot take a full block, so ignore
                if block_starts[1][j] <= self.width - block_size:
                    indices_by_row[row].append(block_starts[1][j]*3)
            self.block_index_sets.append(indices_by_row)
        # flatten rgb channels into each row   [r1,g1,b1,r2,g2,b2,...]
        self.original_values = self.original_values.reshape(self.height, -1)
        self.width = self.original_values.shape[1]

    def get_best_matches(self):
        """
        Description: This method is used to get the best matches to the original values of the differences between
        row pixel values. At every row, the compressed values are compared to the original values to create the new
        diffs so that the error is minimized.
        It iterates through each row, greedily trying to fill the larger block size matches first.
        A mask is used to keep track of where matches have been found as the block sizes get smaller.

        """
        for row_idx in range(1, self.height):
            unfilled_indexes = []
            block_size = 0
            a_set = None
            self.row_idx = row_idx
            filled_mask = np.zeros(self.width)
            row_diffs = self.compressed_values[row_idx - 1, :] - self.original_values[row_idx, :]
            row_queue = np.zeros(self.width)
            self.row_lookup_indexes = {}
            for i, indexes in enumerate(self.block_index_sets):
                col_indexes = indexes[row_idx-1]
                if col_indexes:
                    a_set = self.lookup_table.set_list[i]
                    block_size = a_set.block_size
                    # get the column indexes whose blocks won't overlap with any other previously added blocks
                    unfilled_indexes = check_for_overlap(filled_mask, np.array(col_indexes), block_size)
                if col_indexes and unfilled_indexes:
                    # get the diffs for the valid indexes in the row
                    diffs = get_diffs(row_diffs, np.array(unfilled_indexes), block_size)
                    # find the closest matches for the diffs
                    distances, vector_indexes = self.trees[i].query(diffs)
                    row_matches = np.array([a_set.vectors[index] for index in vector_indexes])
                    row_match_dict = {unfilled_indexes[m]: row_matches[m] for m in range(len(unfilled_indexes))}
                    # check to see if any are under the error threshold
                    # indexes_under_threshold is a list of the indexes from distances/indices/valid_indexes/diffs/row_matches where the error is under the threshold
                    indexes_under_threshold = np.where(distances < self.error_threshold)[0]
                    if len(indexes_under_threshold) > 0:
                        # get the column indexes of the matches where the error is under the threshold
                        under_thresh_col_indexes = np.array(unfilled_indexes)[indexes_under_threshold]
                        # get the matches for the indexes where the error is under the threshold
                        cur_cutoff = 0
                        non_overlapping_indexes = []
                        for k, idx in enumerate(under_thresh_col_indexes):
                            if idx < cur_cutoff:
                                pass
                            else:
                                non_overlapping_indexes.append(k)
                                cur_cutoff = idx + block_size - 1
                        # get the column indexes of the matches where the blocks don't overlap.
                        accepted_col_indexes = under_thresh_col_indexes[non_overlapping_indexes]
                        # get the matches for the accepted column indexes
                        accepted_matches = np.array([row_match_dict[n] for n in accepted_col_indexes])
                        self.add_lookup_indexes(accepted_matches, accepted_col_indexes)
                        # add the matches to the row_queue
                        row_queue = add_matches(row_queue, np.array(accepted_matches), np.array(accepted_col_indexes), block_size)
                        self.match_counter[i] += len(accepted_col_indexes)
                        # add the indexes to the mask
                        filled_mask = add_to_mask(filled_mask, accepted_col_indexes, block_size)
            # fill the rest of the row with the single pixel matches
            # get remaining indexes
            remaining_indexes = np.where(filled_mask == 0)[0]
            # every channel has an index in the row
            # select every third index in remaining_indexes starting with the first index
            # to find the values for a single pixel
            remaining_indexes = remaining_indexes[::3]
            diffs = get_diffs(row_diffs, remaining_indexes, 3)
            if diffs:
                distances, indices = self.trees[-1].query(diffs)
                row_matches = np.array([self.lookup_table.set_list[-1].vectors[index] for index in indices])
                self.add_lookup_indexes(row_matches, remaining_indexes)
                self.match_counter[-2] += len(remaining_indexes)
                # add to row queue
                row_queue = add_matches(row_queue, np.array(row_matches), np.array(remaining_indexes), 3)
            # sort self.row_lookup_indexes keys and add the values to self.compressed_indexes
            self.compressed_indexes += [self.row_lookup_indexes[key] for key in sorted(self.row_lookup_indexes.keys())]
            self.compressed_values[row_idx, :] = np.clip(self.compressed_values[row_idx - 1, :] - row_queue, self.clip_min, self.clip_max)

    def add_lookup_indexes(self, matches, col_indexes):
        lookup_indexes = [self.lookup_table.index_dict[tuple(x)] for x in matches]
        for j in range(len(lookup_indexes)):
            self.row_lookup_indexes[col_indexes[j]] = lookup_indexes[j]

    def paeth_predictor(self, left, above, upper_left):
        """
        Description: This method is used to calculate the paeth value for a given vector.
        :param left: left vector
        :param above: above vector
        :param upper_left: upper left vector
        :return: the paeth value
        """
        p = left + above - upper_left
        pa = abs(p - left)
        pb = abs(p - above)
        pc = abs(p - upper_left)
        return np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, above, upper_left))

    def find_special_match(self, col_idx):

        # returns the special match if one is under the error threshold, otherwise returns an empty array
        # find the paeth, average and left
        left = self.compressed_values[self.row_idx, col_idx - 3:col_idx]
        above = self.compressed_values[self.row_idx - 1, col_idx:col_idx + 3]
        above_left = self.compressed_values[self.row_idx - 1, col_idx - 3:col_idx]
        current_pixel = self.original_values[self.row_idx, col_idx:col_idx + 3]
        above_and_left = left + above
        paeth = self.paeth_predictor(left, above, above_left)
        average = above_and_left // 2
        specials = [paeth, average, left]
        special_errors = np.linalg.norm(specials - current_pixel, axis=1)
        # find the smallest error and its index
        min_error_idx = np.argmin(special_errors)
        min_error = special_errors[min_error_idx]
        if min_error < self.error_threshold:
            self.specials[min_error_idx][0] += 1
            closest_index = min_error_idx
            self.compressed_indexes.append(closest_index)
            adjusted_match = above - specials[min_error_idx]
            return adjusted_match
        else:
            return np.array([])

    def huff_compress(self):
        """
        Description: compress the first bits of values using Huffman compression
        some bits may be uncompressed if the uncompressed bit size is not zero
        """
        values = self.matches
        # indexes = [self.lookup_table.index_dict[tuple(x)] for x in values]
        indexes = self.compressed_indexes
        decompressed_vectors = [self.lookup_table.index_dict_reverse[x] for x in indexes]
        #print("decompressed vectors:", decompressed_vectors)
        indexes = np.array(indexes)
        max_index = self.lookup_table.max_index
        # get the number of bits needed to represent the largest index
        num_bits = len(bin(max_index)[2:])
        remaining_bits = ''
        if self.max_compressed_bit_size < num_bits:
            compressed_bit_size = self.max_compressed_bit_size
            self.uncompressed_bit_size = num_bits - compressed_bit_size
            # if the number of bits needed to represent the largest index is greater than the max compressed bit length
            # when we will need to deal with uncompressed bits
            # get the remaining bits (the least significant bits) and add them to a new list
            remaining_bits = [bin(x)[2:].zfill(num_bits)[compressed_bit_size:] for x in indexes]
            # convert the remaining bits to a single binary string
            remaining_bits = ''.join(remaining_bits)
        else:
            compressed_bit_size = num_bits
        # convert the indexes to bits and take the first bit_size bits of each index
        # (the first bit_size bits are the most significant bits)
        # pad the bits with 0s if necessary to make them all the same length
        compression_bits = [bin(x)[2:].zfill(num_bits)[:compressed_bit_size] for x in indexes]
        # The huffman class expects decimal values, so convert the bits to decimal
        diffs_decimals = [int(x, 2) for x in compression_bits]
        huff = HuffmanCoding()
        huff_compressed, encoded_list = huff.compress(diffs_decimals)
        self.compressed_length = len(huff_compressed)
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
        if col_idx == 0:
            # if we're at the beginning of the row, use the default row value
            values = [self.DEFAULT_ROW_VALUE] * 3
        else:
            values = destination_values[row_idx, col_idx - 3:col_idx].tolist()  # get the previous three values
        # fill neighbors with values from previous row, values is a list of three values that are repeated to fill
        approximate_neighbors = [values[j] for i in range(max_block_size // 3) for j in range(3)]
        destination_values[row_idx - 1, col_idx:col_idx + max_block_size] = approximate_neighbors

    def decompress(self, file_name=''):
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
        compressed_bit_length = header_values['length']
        uncompressed_bit_size = header_values['uncompressed_bit_size']
        first_bits = full_file[header_length:header_length + compressed_bit_length]
        huff = HuffmanCoding()
        first_bits = huff.decompress_file(first_bits)
        uncompressed_bits = full_file[header_length + compressed_bit_length:]
        decompressed_indexes = first_bits
        if uncompressed_bits:
            compressed_bit_size = len(bin(max(first_bits))[2:])
            # convert the integers to bits and pad them with 0s if necessary to make them all the same length (bit_size)
            decompressed_first_bits = [bin(x)[2:].zfill(compressed_bit_size) for x in first_bits]
            # break the remaining bits into groups of remaining_match_bit_size bits
            uncompressed_bits = [uncompressed_bits[i:i + uncompressed_bit_size] for i in
                                 range(0, len(uncompressed_bits), uncompressed_bit_size)]
            # combine decompressed_first_bits and remaining_matches_bits into a single list
            decompressed = [decompressed_first_bits[i] + uncompressed_bits[i] for i in
                            range(len(decompressed_first_bits))]
            # convert to decimal
            decompressed_indexes = [int(x, 2) for x in decompressed]
        # convert to vectors
        decompressed_vectors = [lookup_table.index_dict_reverse[x] for x in decompressed_indexes]

        return decompressed_vectors, decompressed_indexes

    def rebuild_image(self, matches):
        """
        Description: This method is used to rebuild the image.

        :param matches: The list of matches that will be used to rebuild the image.
        """
        decompressed_values = np.zeros((self.height, self.width), dtype=np.int32)
        decompressed_values = np.insert(decompressed_values, 0, self.DEFAULT_ROW_VALUE, axis=0)
        match_idx = 0
        for row_idx in range(1, decompressed_values.shape[0]):
            # get the matches for the row
            col_idx = 0
            while col_idx < self.width:
                #if row_idx == 1 and col_idx > 0:
                    #self.approximate_top_values(row_idx, col_idx, decompressed_values)
                match = matches[match_idx]
                if len(match) == 1:
                    # if the match is a special value, calculate the special value from surrounding values
                    block_size = 3
                    above = decompressed_values[row_idx - 1, col_idx:col_idx + 3]
                    left = decompressed_values[row_idx, col_idx - 3:col_idx]
                    above_left = decompressed_values[row_idx - 1, col_idx - 3:col_idx]
                    paeth = self.paeth_predictor(left, above, above_left)
                    average = (left + above) // 2
                    specials = [paeth, average, left]
                    match = above - specials[match[0]]
                else:
                    block_size = len(match)
                decompressed_values[row_idx, col_idx:col_idx + block_size] = np.clip(decompressed_values[row_idx - 1,
                                                                                     col_idx:col_idx + block_size] - np.array(match),
                                                                                     self.clip_min, self.clip_max)
                col_idx += block_size
                match_idx += 1
        # delete the first row of decompressed_values
        decompressed_values = np.delete(decompressed_values, 0, axis=0)
        # reshape to 3 channels for image display
        self.decompressed_values = decompressed_values.reshape(decompressed_values.shape[0],
                                                               decompressed_values.shape[1] // 3, 3)


@jit(nopython=True)
def check_for_overlap(mask_1, indexes, block_size):
    valid_indexes = []
    for idx in indexes:
        # Ensure that none of the indexes in the block are already in the mask
        if not np.any(mask_1[idx:idx + block_size]):
            valid_indexes.append(idx)
    return valid_indexes


#@jit(nopython=True)
def get_diffs(row, indexes, block_size):
    row = row.flatten()
    diffs = []
    for idx in indexes:
        diffs.append(row[idx:idx + block_size])
    return diffs


#@jit(nopython=True)
def add_to_mask(mask, indexes, block_size):
    for idx in indexes:
        mask[idx:idx + block_size] = True
    return mask


#@jit(nopython=True)
def add_matches(row, matches, indexes, block_size):
    # indexes are the column indexes of the matches
    # matches are the vector matches for the block_size
    for i, idx in enumerate(indexes):
        #match = matches[i*block_size:i*block_size+block_size]
        row[idx:idx + block_size] = matches[i]
    return row

"""
        self.original_values = [[[0,  0,  0], [0, 0, 0], [0,  0,  0], [0, 0, 0], [0,  0,  0], [0, 0, 0], [0,  0,  0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0,  0,  0], [0, 0, 0], [0,  0,  0], [0, 0, 0], [0,  0,  0], [0, 0, 0], [0,  0,  0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
"""