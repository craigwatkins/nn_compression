import numpy as np
import cv2 as cv
import time
from numba import jit


from huffman_compression import HuffmanCoding
import padded_binary as pb
from header import Header
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from canonical_huffman import HuffmanCoding as CanonicalHuffmanCoding
class NNCompressor:
    """
    Description: This class is used to compress and decompress a given image.
    """

    DEFAULT_ROW_VALUE = 128

    def __init__(self):
        """
        Description: This method is used to initialize the class variables.
        """
        self.lookup_table, self.grouped, self.row_centroids = self.get_lookup_table()
        #self.row_tree = KDTree(self.row_centroids)
        self.grouped_rev = {v: k for k, v in self.grouped.items()}
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
        self.search_depth = 1
        self.decompressed_values = None
        self.max_compressed_bit_size = 16
        self.compressed_length = 0
        self.uncompressed_bit_size = 0
        self.compressed_path = ''
        self.row_idx = 0
        self.specials = [[0], [0], [0]]
        self.row_matches = []
        self.block_index_sets = []
        self.row_lookup_indexes = {}
        self.transposed = False
        self.custom_tree = None
        self.common_dict = {}
        self.custom_dict_rev = {}
        self.top_left_mask = None

    def get_lookup_table(self):
        """
        Description: This method is used to get the lookup tables.
        """
        from create_lookup_tables import create_lookup_table

        return create_lookup_table()

    def compress(self, source_path, compressed_path, error_threshold=2, search_depth=5):
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
        print("getting samples")
        sample_size = 1
        samples = sample_diffs('image_avg_diffs.db', num_images=8090, num_samples=30, sample_size=sample_size)
        samples = samples.reshape(-1, sample_size * 3)  # flatten the last dimension
        samples = set([tuple(x) for x in samples])
        samples = list(samples)
        samples = np.array(samples)
        # remove all entries where the sum of the absolute values is more than 15
        samples = samples[np.sum(np.abs(samples), axis=1) < 30]
        a_set = self.lookup_table.set_list[-1]
        distance, indices = a_set.get_matches(samples)
        self.common_dict = {tuple(samples[i]): indices[i] for i in range(len(samples))}
        print("common dict length:", len(self.common_dict))

        self.search_depth = search_depth
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
        print("match counter:", self.match_counter)
        # remove the top row of compressed_values, it's not actually part of the image
        self.compressed_values = np.delete(self.compressed_values, 0, axis=0)
        self.original_values = np.delete(self.original_values, 0, axis=0)
        # remove the left columns of compressed_values, it's not actually part of the image
        self.compressed_values = np.delete(self.compressed_values, np.s_[:3], axis=1)
        self.original_values = np.delete(self.original_values, np.s_[:3], axis=1)

        image_data = self.huff_compress()
        header_obj = Header()
        header = header_obj.build_header([self.compressed_values.shape[0], self.compressed_values.shape[1],
                                          self.compressed_length, self.uncompressed_bit_size,
                                          self.clip_min, self.clip_max,self.transposed])
        # save the compressed image
        pb.write_padded_bytes(header + image_data, self.compressed_path)
        # print("match counter:", self.match_counter)

        # reshape compressed values for image display
        self.compressed_values = self.compressed_values.reshape(self.compressed_values.shape[0],
                                                                self.compressed_values.shape[1] // 3, 3)
        # transpose if necessary
        if self.transposed:
            self.compressed_values = np.transpose(self.compressed_values, (1, 0, 2))

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

        top_diffs = self.original_values[1:] - self.original_values[:-1]
        top_diffs_2 = self.original_values[1:, 1:] - self.original_values[:-1, 1:]
        left_diffs = self.original_values[1:, 1:] - self.original_values[1:, :-1]

        top_diffs_2 = top_diffs_2.reshape(-1, 3)
        left_diffs = left_diffs.reshape(-1, 3)
        # for each row find the sum of the absolutes values of top_diffs_2 and left_diffs
        top_diffs_2 = np.sum(np.abs(top_diffs_2), axis=1)
        left_diffs = np.sum(np.abs(left_diffs), axis=1)
        mask = np.abs(left_diffs) < np.abs(top_diffs_2)
        # get the counts for the number of occurrences of '00', '01', '10', '11' in mask
        # group the mask into 2 bit numbers
        mask_2bit = mask.reshape(-1, 2)
        # convert each 2 bit number in mask to a decimal number
        mask_decimals = mask_2bit.dot([1, 2])
        mask_2d = mask.reshape(self.original_values.shape[0] - 1, -1)
        self.top_left_mask = mask_2d

        huff = HuffmanCoding()
        huff_compressed, encoded_list = huff.compress(mask_decimals)
        print("huff mask dict size:", huff.dict_byte_size)
        print("huff mask compressed size:", len(huff_compressed)//8)

        """
       
        avg_diffs = (top_diffs_2 + left_diffs) // 2
        avg_diffs = avg_diffs.reshape(-1, 3)
        avg_diffs = avg_diffs.tolist()
        avg_diffs = [tuple(x) for x in avg_diffs]
        from collections import Counter
        counts = Counter(avg_diffs)
        # sort the counts by the number of occurrences
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        # get the most common pairs
        most_common = sorted_counts[:256]
        print("most common", most_common)
        print("number of occurences of the top samples", sum([x[1] for x in most_common]))
        """
        self.height = self.original_values.shape[0]
        self.width = self.original_values.shape[1]
        """
        top_diffs_flat = top_diffs.reshape(top_diffs.shape[0]*top_diffs.shape[1], -1)
        top_diff_sizes = np.linalg.norm(top_diffs_flat, axis=1)
        threshold_mults = [3, 3, 3, 3, self.search_depth, self.search_depth, self.search_depth]
        matches = [0]*(len(self.lookup_table.set_list) - 1)
        """
        """
        Try to find runs of small diffs appropriate for each block size across the entire image
        The heuristic is to sum the Euclidean distances of the r, g, b diffs in a block, and if the sum 
        is below a threshold, then the block is a candidate for matching.
        Record the starting indexes of the blocks that are below the threshold by block size and by row
        """
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
        """
        # flatten rgb channels into each row   [r1,g1,b1,r2,g2,b2,...]
        self.original_values = self.original_values.reshape(self.height, -1)
        self.width = self.original_values.shape[1]

    def get_best_matches_old(self):
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
                    unfilled_indexes = check_if_filled(filled_mask, np.array(col_indexes), block_size)
                if col_indexes and unfilled_indexes:
                    # get the diffs for the valid indexes in the row
                    diffs = get_diffs(row_diffs.flatten(), np.array(unfilled_indexes), block_size)
                    # find the closest matches for the diffs
                    distances, vector_indexes = a_set.get_matches(diffs)
                    row_matches = np.array([a_set.vectors[index] for index in vector_indexes])
                    row_match_dict = {unfilled_indexes[m]: row_matches[m] for m in range(len(unfilled_indexes))}
                    # check to see if any are under the error threshold
                    # indexes_under_threshold is a list of the indexes from distances where error is under threshold
                    indexes_under_threshold = np.where(distances < self.error_threshold)[0]
                    if len(indexes_under_threshold) > 0:
                        # get the column indexes of the matches where the error is under threshold
                        under_thresh_col_indexes = np.array(unfilled_indexes)[indexes_under_threshold]
                        # remove indexes with overlapping blocks
                        non_overlapping_indexes = get_non_overlapping(under_thresh_col_indexes, block_size)
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
            remaining_indexes = np.where(filled_mask == 0)[0]
            # get the first index of every r,g,b pixel
            remaining_indexes = remaining_indexes[::3]
            diffs = list(get_diffs(row_diffs.flatten(), remaining_indexes, 3))
            if diffs:
                a_set = self.lookup_table.set_list[-1]
                distances, indices = a_set.get_matches(diffs)
                row_matches = np.array([a_set.vectors[index] for index in indices])
                self.add_lookup_indexes(row_matches, remaining_indexes)
                self.match_counter[-2] += len(remaining_indexes)
                row_queue = add_matches(row_queue, np.array(row_matches), np.array(remaining_indexes), 3)
            # sort by column index and add the vector indexes in the correct order for compression
            self.compressed_indexes += [self.row_lookup_indexes[key] for key in sorted(self.row_lookup_indexes.keys())]
            # add the row_queue to the compressed values
            self.compressed_values[row_idx, :] = np.clip(self.compressed_values[row_idx - 1, :] - row_queue, self.clip_min, self.clip_max)

    def get_best_matches(self):
        import time
        start = time.time()
        common_matches = 10
        a_set = self.lookup_table.set_list[0]
        a_set2 = self.lookup_table.set_list[-1]
        max_index = self.lookup_table.max_index
        errors = []
        diff_sizes = []

        for row_idx in range(1, self.height):
            for col_idx in range(3, self.width, 3):
                left = self.compressed_values[row_idx, col_idx - 3:col_idx]
                current = self.original_values[row_idx, col_idx:col_idx + 3]
                top = self.compressed_values[row_idx - 1, col_idx:col_idx + 3]
                avg = (top + left) // 2
                #diffs = avg - current
                """
                if self.top_left_mask[row_idx - 1, col_idx // 3 - 1] == 0:
                    diffs = top - current
                    reference = top
                else:
                    diffs = left - current
                    reference = left
                """
                diffs = avg - current
                reference = avg
                distance, index = a_set.get_matches(diffs)
                error_threshold = self.error_threshold
                errors.append(a_set.vectors[index] - diffs)
                if distance < error_threshold:
                    match = a_set.vectors[index]
                    self.compressed_indexes.append(self.lookup_table.index_dict[match])
                    self.compressed_values[row_idx, col_idx:col_idx + 3] = np.clip(reference - match, self.clip_min,
                                                                                   self.clip_max)
                else:
                    distance, index = a_set2.get_matches(diffs)
                    match = a_set2.vectors[index]
                    self.compressed_indexes.append(self.lookup_table.index_dict[match])
                    self.compressed_values[row_idx, col_idx:col_idx + 3] = np.clip(reference - match, self.clip_min,
                                                                                   self.clip_max)
                self.matches.append(match)

                #distance, index = self.custom_tree.query(diffs)
                #match = self.custom_dict_rev[index]
                #self.compressed_indexes.append(index)

                """
                new_diffs = diffs - match
                distance, index = a_set2.get_matches(new_diffs)
                new_match = np.array(a_set2.vectors[index])
                adj_diffs = diffs - (np.array(match) + new_match)
                if np.linalg.norm(adj_diffs) < np.linalg.norm(new_diffs):
                    self.compressed_indexes.append(self.lookup_table.index_dict[tuple(new_match)])
                    match = match + new_match
                else:
                    self.compressed_indexes.append(self.lookup_table.index_dict[(0, 0, 0)])
                """

        stop = time.time()
        print("time:", stop - start)
        print("common matches:", common_matches)
        self.huff_errors(errors)
        """
        # plot 2d graph of errors vs diffs size
        import matplotlib.pyplot as plt
        plt.scatter(diff_sizes, errors)
        # label plot
        plt.xlabel('diffs size')
        plt.ylabel('error')
        plt.show()
        # plot histogram of sum of errors binned by diffs size

        plt.figure()
        plt.hist(diff_sizes, bins=100, weights=errors)
        plt.xlabel('diffs size')
        plt.ylabel('sum of errors')
        plt.show()
        """
    def huff_errors(self, errors):
        # get the unique errors and assign each of them an index
        # then compress the indexes with huffman
        unique_errors = set([tuple(x) for x in errors])
        new_index_dict = {value: index for index, value in enumerate(unique_errors)}

        new_indexes = [new_index_dict[tuple(x)] for x in errors]
        # compress the new indexes with huffman
        huff = HuffmanCoding()
        huff_compressed, encoded_list = huff.compress(new_indexes)
        # get the size of the dictionary
        dict_size = huff.dict_byte_size
        # get the size of the compressed data
        compressed_size = len(huff_compressed)
        print("error dict size:", dict_size)
        print("error compressed size:", compressed_size//8)

    def add_lookup_indexes(self, matches, col_indexes):
        # adds the lookup indexes for the matches to self.row_lookup_indexes
        lookup_indexes = [self.lookup_table.index_dict[tuple(x)] for x in matches]
        for j in range(len(lookup_indexes)):
            self.row_lookup_indexes[col_indexes[j]] = lookup_indexes[j]

    def get_centroids(self, k, samples):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(samples)
        # Calculate centroids
        centroids = kmeans.cluster_centers_
        return centroids

    def get_row_distribution(self, row, num_centroids):
        group_counter = Counter(row)
        sample_size = len(row)
        centroid_table = [0] * num_centroids
        for key, value in group_counter.items():
            centroid_table[key] = value / sample_size

        return centroid_table

    def huff_compress(self):
        """
        Description: compress the first bits of values using Huffman compression
        some bits may be uncompressed if the uncompressed bit size is not zero
        """
        values = self.matches
        # indexes = [self.lookup_table.index_dict[tuple(x)] for x in values]
        indexes = self.compressed_indexes
        indexes = np.array(indexes)
        # get the number of occurrences for each index
        unique, counts = np.unique(indexes, return_counts=True)
        # create a dictionary of the indexes and their counts
        index_dict = dict(zip(unique, counts))
        # sort the dictionary by the counts
        sorted_index_dict = {k: v for k, v in sorted(index_dict.items(), key=lambda item: item[1], reverse=True)}
        print("sorted index dict", sorted_index_dict)
        new_indexes = []
        start = 0
        match_length = 1
        last_match = None
        match_lens = [0, 0, 0, 0, 0, 0]
        starts = []
        max_match = 5
        indexes_len = len(indexes)

        while start < indexes_len - 9999999999:
            last_match = indexes[start]
            match_length = 1
            for i in range(2, max_match + 1):
                potential_match = tuple(indexes[start:start + i])
                match = self.grouped.get(potential_match, None)
                if match is not None:
                    last_match = match
                    match_length = i
                if i == max_match or start + i == indexes_len:
                    new_indexes.append(last_match)
                    start += match_length
                    match_lens[match_length - 1] += 1
                    break

        a_set = self.lookup_table.set_list[0]

        block_size = 512
        total_sum = np.sum(indexes)
        num_blocks = len(indexes) / block_size
        average_block_size = total_sum/num_blocks
        small_blocks = []
        large_blocks = []
        for i in range(0, len(indexes), block_size):
            block = indexes[i:i+block_size]
            block_sum = sum(block)
            if block_sum < average_block_size:
                small_blocks.append(block)
            else:
                large_blocks.append(block)
        # flatten small_blocks and large_blocks
        small_blocks = [item for sublist in small_blocks for item in sublist]
        large_blocks = [item for sublist in large_blocks for item in sublist]


        huff_small = HuffmanCoding()
        huff_large = HuffmanCoding()
        huff_small_compressed, encoded_list_small = huff_small.compress(small_blocks)
        huff_large_compressed, encoded_list_large = huff_large.compress(large_blocks)
        print("small huff size:", len(huff_small_compressed)//8)
        print("large huff size:", len(huff_large_compressed)//8)
        print("split cost ", num_blocks//8)





        print("match lengths:", match_lens)
        print("indexes length:", len(indexes))
        # indexes = new_indexes
        print("new indexes length:", len(indexes))
        print("unique indexes:", len(set(indexes)))

        huff = HuffmanCoding()
        huff_compressed, encoded_list = huff.compress(indexes)
        self.compressed_length = len(huff_compressed)
        file_info = huff_compressed
        print("actual dict size:", huff.dict_byte_size)

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
        self.transposed = header_values['transposed']
        compressed_bit_length = header_values['length']
        uncompressed_bit_size = header_values['uncompressed_bit_size']
        first_bits = full_file[header_length:header_length + compressed_bit_length]
        huff = HuffmanCoding()
        decompressed_indexes = huff.decompress_file(first_bits)
        uncompressed_bits = full_file[header_length + compressed_bit_length:]

        """
        lookup_table.index_dict_reverse.update(self.grouped_rev)
        decompressed_indexes2 = []
      
        for i, index in enumerate(decompressed_indexes):
            if index in self.grouped_rev:
                a_tuple = self.grouped_rev[index]
                for item in a_tuple:
                    decompressed_indexes2.append(item)
            else:
                decompressed_indexes2.append(index)
        """
        # convert to vectors
        decompressed_vectors = [lookup_table.index_dict_reverse[x] for x in decompressed_indexes]

        return decompressed_vectors, decompressed_indexes

    def rebuild_image_old(self, matches):
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
        # transpose if necessary
        if self.transposed:
            self.decompressed_values = np.transpose(self.decompressed_values, (1, 0, 2))

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
        # transpose if necessary
        if self.transposed:
            self.decompressed_values = np.transpose(self.decompressed_values, (1, 0, 2))


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


def get_centroids(k, samples):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(samples)
    # Calculate centroids
    centroids = kmeans.cluster_centers_
    return centroids


def sample_diffs(db_name, num_images, num_samples, sample_size=1):
    import sqlite3
    import random
    SEED = 42

    height = 127
    width = 127
    # set random seed
    random.seed(SEED)
    # Connect to the database
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Retrieve the diffs for a specified number of images
    cursor.execute("SELECT diffs FROM images LIMIT ?", (num_images,))
    rows = cursor.fetchall()

    all_samples = []

    for row in rows:
        # Reshape the diff data to its original shape
        diff_array = np.frombuffer(row[0], dtype=np.int16).reshape(height, width, 3)

        for _ in range(num_samples):
            # Randomly select a row
            row_idx = random.randint(0, height-1)
            # Ensure sample doesn't cross rows
            if 128 - sample_size > 0:
                start_col_idx = random.randint(0, width - sample_size)
                sample = diff_array[row_idx, start_col_idx:start_col_idx + sample_size, :]
                all_samples.append(sample)

    connection.close()
    return np.array(all_samples)



"""
            row_distribution = self.get_row_distribution(row, len(a_set.vectors))
            row_distance, row_index = self.row_tree.query(row_distribution)
            row_match = self.row_centroids[row_index]
            row_match_dict = {i: row_match[i] for i in range(len(row_match))}
            # compress each row with huffman
            huff_fixed = CanonicalHuffmanCoding()
            huff_fixed.huff_dict.make_dictionary(occurrences=row_match_dict)
            huff_fixed.compress(row, fixed_dict=True)
            #huff_compressed, encoded_list = huff.compress(row)
            total_row_size += len(huff_fixed.encoded_text)//8
            #huff_dict_size += huff.dict_byte_size


"""

"""
index_rows = np.reshape(indexes, (-1, self.height-1))
total_row_size = 0
huff_dict_size = 0
row_increment = 1
row_sums = np.sum(index_rows, axis=1)
for i in range(0, len(index_rows), row_increment):
    row = index_rows[i:i+row_increment]
    row = row.flatten()
    total_row_size += sum(row)
average_row_size = total_row_size/(len(index_rows))
small_rows = []
large_rows = []
for i in range(0, len(index_rows), row_increment):
    row = index_rows[i:i+row_increment]
    row = row.flatten()
    row_sum = sum(row)
    if row_sum < average_row_size*.97:
        small_rows.append(row)
    else:
          large_rows.append(row)
# flatten small_rows
small_rows = np.array(small_rows)
large_rows = np.array(large_rows)

huff_small = HuffmanCoding()
huff_large = HuffmanCoding()
huff_small_compressed, encoded_list_small = huff_small.compress(small_rows.flatten())
huff_large_compressed, encoded_list_large = huff_large.compress(large_rows.flatten())
print("small huff size:", len(huff_small_compressed)//8)
print("large huff size:", len(huff_large_compressed)//8)
"""