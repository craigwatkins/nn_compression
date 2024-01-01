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
        """
       
       
        image_data = self.huff_compress()
        header_obj = Header()
        header = header_obj.build_header([self.compressed_values.shape[0], self.compressed_values.shape[1],
                                          self.compressed_length, self.uncompressed_bit_size,
                                          self.clip_min, self.clip_max])
        # save the compressed image
        pb.write_padded_bytes(header + image_data, self.compressed_path)
        print("match counter:", self.match_counter)
        # reshape compressed values for image display
        #self.compressed_values = self.compressed_values.reshape(self.compressed_values.shape[0],
                                                                #self.compressed_values.shape[1] // 3, 3)
        """

        return self.compressed_values

    def preprocess_image(self, source_path):
        self.original_values = cv.imread(source_path).astype(np.int16)
        # insert a row of default values at the top of the image
        self.original_values = np.insert(self.original_values, 0, self.DEFAULT_ROW_VALUE, axis=0)
        self.height = self.original_values.shape[0]
        self.width = self.original_values.shape[1]
        top_diffs = self.original_values[1:] - self.original_values[:-1]
        top_diff_sizes = np.linalg.norm(top_diffs, axis=2).flatten()
        # reshape to a list of r,g,b vectors
        #top_diffs = top_diffs.reshape(top_diffs.shape[0]*top_diffs.shape[1], -1)
        threshold_mults = [4, 4, 3, 3, 3, 3, 3, 2]
        matches = [0]*(len(self.lookup_table.set_list) - 1)

        for i in range(len(self.lookup_table.set_list)-1):
            indices_by_row = [[] for _ in range(self.height)]
            a_set = self.lookup_table.set_list[i]
            block_size = a_set.block_size//3
            threshold = threshold_mults[i] * self.error_threshold
            kernel = np.ones(block_size, dtype=int)
            padding_needed = len(kernel) - 1
            top_diff_sizes_convolved = np.convolve(top_diff_sizes, kernel, 'valid')
            top_diff_sizes_convolved = np.pad(top_diff_sizes_convolved, (0, padding_needed), mode='constant', constant_values=99999).reshape(self.height-1, self.width)
            # find the starting indices of blocks where the convolution result is below threshold
            block_starts = np.where(top_diff_sizes_convolved < threshold)
            matches[i] = len(block_starts[0])
            for j, row in enumerate(block_starts[0]):
                # the end of a row does not have valid values from the kernel, and cannot take a full block, so ignore
                if block_starts[1][j] <= self.width - block_size:
                    indices_by_row[row].append(block_starts[1][j])
            self.block_index_sets.append(indices_by_row)

        # flatten rgb channels into each row   [r1,g1,b1,r2,g2,b2,...]
        #combined_channels = img.reshape(img.shape[0], -1)
        # add an extra row at the top of combined_channels
        # this aligns it with the rows in the compressed_values array
        #combined_channels = np.insert(combined_channels, 0, self.DEFAULT_ROW_VALUE, axis=0)
        # convert original values to avoid overflow errors
        #self.original_values = combined_channels.astype(np.int16)

    def get_best_matches_old(self):
        """
        Description: This method is used to iterate over the image and find the difference between the actual values of
        the current row and the approximated values of the previous row. The differences are approximated by finding the closest match in the lookup table.
        The closest match is then used to generate the new values for the current row which is then stored in the
        compressed_values array.
        """
        max_block_size = self.lookup_table.max_block_size
        for row_idx in range(1, self.height):
            self.row_idx = row_idx
            col_idx = 0
            row_diffs = self.compressed_values[row_idx - 1, :] - self.original_values[row_idx, :]
            if row_idx > 1:
                self.get_row_matches(row_diffs)
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

    def get_best_matches(self):
        for row_idx in range(1, self.height):
            self.row_idx = row_idx
            reserved_mask = np.zeros(self.width)
            row_diffs = self.compressed_values[row_idx - 1, :] - self.original_values[row_idx, :]
            row_queue = np.zeros((self.width, 3))
            for i, indexes in enumerate(self.block_index_sets):
                col_indexes = indexes[row_idx]
                if col_indexes:
                    new_mask = np.zeros(self.width)
                    a_set = self.lookup_table.set_list[i]
                    block_size = a_set.block_size//3
                    # get the column indexes whose blocks won't overlap with any other previously added blocks
                    valid_indexes = check_for_overlap(reserved_mask, np.array(col_indexes), block_size)
                    # get the diffs for the valid indexes in the row
                    diffs = get_diffs(row_diffs, np.array(valid_indexes), block_size)
                    # find the closest matches for the diffs
                    distances, indices = self.trees[i].query(diffs)
                    row_matches = np.array([a_set.vectors[index] for index in indices])
                    # check to see if any are under the error threshold
                    # indexes_under_threshold is a list of the indexes from distances/indices/valid_indexes/diffs/row_matches where the error is under the threshold
                    indexes_under_threshold = np.where(distances < self.error_threshold)[0]
                    if indexes_under_threshold.any():
                        # get the column indexes of the matches where the error is under the threshold
                        under_thresh_col_indexes = np.array(valid_indexes)[indexes_under_threshold]
                        # get the matches for the indexes where the error is under the threshold
                        row_matches = row_matches[indexes_under_threshold]
                        # check for overlap within the accepted column indexes
                        col_index_diffs = under_thresh_col_indexes[1:] - under_thresh_col_indexes[:-1]
                        # get the indexes within col_index_diffs where the difference from the previous index
                        # is greater than the block size (the blocks don't overlap)
                        non_overlapping_indexes = np.where(col_index_diffs > block_size)[0]
                        non_overlapping_indexes = non_overlapping_indexes + 1
                        # the first index of accepted_col_indexes is always valid, so add it to the beginning of the list
                        #non_overlapping_indexes = np.insert(non_overlapping_indexes, 0, 0)
                        # get the column indexes of the matches where the blocks don't overlap.
                        accepted_col_indexes = np.insert(under_thresh_col_indexes[non_overlapping_indexes], 0, under_thresh_col_indexes[0])
                        # get the matches for the accepted column indexes
                        accepted_matches = np.insert(row_matches[non_overlapping_indexes], 0, row_matches[0])
                        accepted_matches = accepted_matches.reshape(accepted_matches.shape[0]//3, 3)
                        # add the matches to the row_queue
                        row_queue = add_matches(row_queue, np.array(accepted_matches), np.array(accepted_col_indexes), block_size)
                        # add the indexes to the mask
                        reserved_mask = add_to_mask(reserved_mask, accepted_col_indexes, block_size)
            # fill the rest of the row with the single pixel matches
            # get remaining indexes
            remaining_indexes = np.where(reserved_mask == 0)[0]
            diffs = get_diffs(row_diffs, remaining_indexes, 1)
            distances, indices = self.trees[-1].query(diffs)
            #self.lookup_table.set_list[-1].vectors
            row_matches = np.array([self.lookup_table.set_list[-1].vectors[index] for index in indices])
            # add to row queue
            row_queue = add_matches(row_queue, np.array(row_matches), np.array(remaining_indexes), 1)
            self.compressed_values[row_idx, :] = np.clip(self.compressed_values[row_idx - 1, :] - row_queue, self.clip_min, self.clip_max)

    def get_row_matches(self, row_diffs):
        """
        Description: This method is used to find the closest match in the lookup table for each pixel
        in the current row (self.row_matches). It uses a KDTree to find the best match.
        :param row_diffs: differences between the actual values of the current row and the
        approximated values of the previous row
        :return: None
        """
        row_diffs = [row_diffs[i:i + 3] for i in range(0, len(row_diffs), 3)]
        distances, indices = self.kd_tree.query(row_diffs)
        # Convert indices to actual points from setB
        self.row_matches = [self.one_pixel_set[index] for index in indices]


    def get_match(self, col_idx, row_diffs):
        """
        Description: This method is used to find the closest match in the lookup table for a given vector.
        It tries to find the largest match possible, starting with the largest block size and moving down
        to the smallest block size.
        :param col_idx: column index of the current pixel
        :param row_diffs: differences between the actual values of the current row
        and the approximated values of the previous row
        :return:
            closest_match: the closest match in the lookup table
            block_size: the size of the block that was matched
        """
        block_size = 0
        closest_match = None
        closest_index = None
        return_the_match = False
        i = 0
        for i, a_set in enumerate(self.lookup_table.set_list):
            block_size = a_set.block_size
            if col_idx + block_size > self.width:
                # if the there isn't enough room for a vector of this size to fit in the row
                # move on to the next smaller block size
                continue
            diff = row_diffs[col_idx:col_idx + block_size]
            # if the block_size is only one pixel (r,g,b), look for special values
            # skip the first row and column to ensure that all values are available
            if block_size == 3 and col_idx > 0 and self.row_idx > 0:
                special_match = self.find_special_match(col_idx)
                if len(special_match) > 0:
                    return special_match, block_size
            if block_size == 3 and self.row_idx > 1:
                # row_matches are only bulk generated for 1 pixel blocks after the first row
                # and here we can simply get the appropriate match from the row_matches list
                closest_match = self.row_matches[col_idx // 3]
                closest_index = self.lookup_table.index_dict[closest_match]
                return_the_match = True
            else:
                # we either have a multi-pixel block or we're in the first row
                closest_match = tuple(a_set.set_index.get_closest_match(diff))
                closest_index = self.lookup_table.index_dict[closest_match]
                if block_size > 3:
                    error = np.linalg.norm(diff - np.array(closest_match))
                    return_the_match = error < self.error_threshold
                else:
                    # we won't find a better match after this, so return the match regardless of error
                    return_the_match = True
            if return_the_match:
                self.match_counter[i] += 1
                self.compressed_indexes.append(closest_index)
                return closest_match, block_size
            else:
                continue

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
        decompressed_values = np.insert(decompressed_values, 0, 128, axis=0)
        match_idx = 0
        for row_idx in range(1, decompressed_values.shape[0]):
            # get the matches for the row
            col_idx = 0
            while col_idx < self.width:
                if row_idx == 1 and col_idx > 0:
                    self.approximate_top_values(row_idx, col_idx, decompressed_values)
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
        if not np.any(mask_1[idx:idx + block_size]):
            valid_indexes.append(idx)
    return valid_indexes


#@jit(nopython=True)
def get_diffs(row, indexes, block_size):
    row = row.flatten()
    diffs = []
    for idx in indexes:
        diffs.append(row[3*idx:3*idx + block_size*3])
    return diffs


#@jit(nopython=True)
def add_to_mask(mask, indexes, block_size):
    for idx in indexes:
        mask[idx:idx + block_size] = True
    return mask


#@jit(nopython=True)
def add_matches(row, matches, indexes, block_size):
    for i, idx in enumerate(indexes):
        match = matches[i*block_size:i*block_size+block_size]
        row[idx:idx + block_size] = match
    return row


