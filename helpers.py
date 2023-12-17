import numpy as np
import cv2
import os
import pickle


def save_as_binary(file_path, file_name, integers):
    from huffman_compression import HuffmanCoding
    import padded_binary as pb
    """
    Save a list of integers as a binary file using Huffman coding.

    Args:
        file_path (str): Path to the directory where the file will be saved.
        file_name (str): Name of the file without extension.
        integers (list): List of integers to be saved.

    Returns:
        int: Size of the saved file in bytes.
    """
    #integers = [int(x) + 256 for x in integers]  # make sure the values are all positive
    huff = HuffmanCoding()
    save_info, encoded_list = huff.compress(integers)
    full_path = file_path + file_name + '.bin'
    pb.write_padded_bytes(save_info, full_path)
    return os.path.getsize(full_path)

def open_binary(file_path, file_name):
    from huffman_compression import HuffmanCoding
    import padded_binary as pb
    """
    Open a binary file and return a list of integers.

    Args:
        file_path (str): Path to the directory where the file is located.
        file_name (str): Name of the file without extension.

    Returns:
        list: List of integers.
    """
    full_path = file_path + file_name + '.bin'
    binaries = pb.read_padded_bytes(full_path)
    if binaries is None:
        print("Error: Could not open file: ", full_path)
        return None
    huff = HuffmanCoding()
    decompressed = huff.decompress_file(binaries)
    #decoded_list = [int(x) - 256 for x in decoded_list]  # make sure the values are all positive
    return decompressed


def save_hsv(h, s, v, filepath):
    # clip the values to between 0 and 255
    v = np.clip(v, 0, 255)
    # convert to uint8
    v = v.astype(np.uint8)
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    # combine h, s, and v into one image
    hsv_image = cv2.merge([h, s, v])
    # convert to uint8
    hsv_image = hsv_image.astype(np.uint8)
    # convert hsv to bgr
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # save the image
    save_image(hsv_image, filepath)


def save_image(pixels, filepath):
    # Use OpenCV's imwrite function to save the image
    success = cv2.imwrite(filepath, pixels)

    if not success:
        raise ValueError(f"Could not save image to the path: {filepath}")

def save_jpeg(pixels, filepath, quality=90):
    # Use OpenCV's imwrite function to save the image
    success = cv2.imwrite(filepath, pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])

    if not success:
        raise ValueError(f"Could not save image to the path: {filepath}")


def extract_hsv(filepath):
    # Open an image and separate it into H, S, V values
    image = cv2.imread(filepath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)
    return H, S, V


def compare_images(filepath1, filepath2):
    # takes two grayscale images as input
    # returns the mean absolute error between them
    # load image 1 and get h,s,v values
    h, s, v = extract_hsv(filepath1)
    # load the images
    #img1 = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
    img1 = v
    img2 = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)
    # crop each image to 512x512 from the top left corner
    img1 = img1[:500, :500]
    img2 = img2[:500, :500]
    # convert the images to numpy arrays
    img1_vals = np.array(img1).flatten()
    img2_vals = np.array(img2).flatten()

    # calculate the mean absolute error between the two images
    mae = np.mean(np.abs(img1_vals - img2_vals))
    print("mean absolute error:", mae)
    return mae


def calculate_psnr(original_path, compressed_path):
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path)
    mse = np.mean((original - compressed) ** 2)
    mae = np.mean(np.abs(original - compressed))
    if mse == 0:  # MSE is zero means no noise is present in the signal.
                  # Therefore PSNR is infinite (perfect fidelity).
        return float('inf')
        #mse = .00001
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



def save_pkl(cluster_centers, filename):
    # save the cluster centers to a pickle file
    with open(filename, "wb") as f:
        pickle.dump(cluster_centers, f)


def find_closest(sorted_list, target):
    # binary search for the closest value to target in sorted_list
    if not sorted_list:
        return None
    low, high = 0, len(sorted_list) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] < target:
            low = mid + 1
        elif sorted_list[mid] > target:
            high = mid - 1
        else:
            return sorted_list[mid]
    if low < len(sorted_list) and high >= 0:
        return sorted_list[low] if abs(sorted_list[low] - target) < abs(sorted_list[high] - target) else sorted_list[high]
    elif low < len(sorted_list):
        return sorted_list[low]
    else:
        return sorted_list[high]






def kmeans_elbow(vectors, range_min, range_max, step=1, random_state=42, n_init=10 ):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    inertia = []
    K_range = range(range_min, range_max, step)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        kmeans.fit(vectors)
        inertia.append(kmeans.inertia_)

    # Step 4: Plot the elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertia, 'o-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(K_range)
    plt.show()
