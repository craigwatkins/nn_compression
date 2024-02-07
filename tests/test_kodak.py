import cv2
import os
import numpy as np

from src import save_jpeg
from src import NNCompressor

"""
This is a script to compress all of the images in the kodak dataset and compare the size of the resulting files to the 
JPEG files. It also calculates the PSNR of the compressed images and the JPEG images to confirm that the quality is
similar.
"""


def get_images(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_list.append((os.path.join(directory, filename), filename))
    return image_list

def compress_image(image_path, save_path, error_threshold):
    compressor = NNCompressor()
    compressed = compressor.compress(image_path, save_path, error_threshold)
    return compressed


def calculate_psnr(original_path, compressed):
    original = cv2.imread(original_path)
    #compressed = cv2.imread(compressed_path)
    mse = np.mean((original - compressed) ** 2)
    mae = np.mean(np.abs(original - compressed))
    if mse == 0:  # MSE is zero means no noise is present in the signal.
                  # Therefore PSNR is infinite (perfect fidelity).
        return float('inf')
        #mse = .00001
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compress_and_compare():
    images = get_images("../images/kodak")
    error_thresholds = [0.0]*25
    error_thresholds[0] = 3.3
    error_thresholds[1] = 20
    error_thresholds[2] = 20
    error_thresholds[3] = 20
    error_thresholds[4] = 8.1
    error_thresholds[5] = 4.8
    error_thresholds[6] = 6.1
    error_thresholds[7] = 6.1
    error_thresholds[8] = 5.4
    error_thresholds[9] = 6.1
    error_thresholds[10] = 4.5
    error_thresholds[11] = 6.7
    error_thresholds[12] = 5
    error_thresholds[13] = 20
    error_thresholds[14] = 20
    error_thresholds[15] = 3.1
    error_thresholds[16] = 4.7
    error_thresholds[17] = 7.6
    error_thresholds[18] = 4.4
    error_thresholds[19] = 20
    error_thresholds[20] = 6.1
    error_thresholds[21] = 20
    error_thresholds[22] = 20
    error_thresholds[23] = 20
    image_settings = [[] for i in range(len(images))]
    # start and stop are used to adjust the range of images to compress at a time for fine-tuning error thresholds
    start = 0
    stop = 23
    for i, image in enumerate(images):
        if i < start:
            continue
        bin_save_path = f"../bins/test/{image[1]}.bin"
        jpeg_file_path = f"../images/test/{image[1]}.jpeg"
        compressed = compress_image(image[0], bin_save_path, error_thresholds[i])
        # save compressed image
        psnr = calculate_psnr(image[0], compressed)
        save_jpeg(cv2.imread(image[0]), jpeg_file_path, 100)
        psnr_jpeg = calculate_psnr(image[0], cv2.imread(jpeg_file_path))
        # get jpeg size
        jpeg_size = os.path.getsize(jpeg_file_path)
        # get compressed size
        compressed_size = os.path.getsize(bin_save_path)
        image_settings[i] = (image[1], psnr, psnr_jpeg, error_thresholds[i], compressed_size, jpeg_size)
        image = image_settings[i]
        # PSNR diff should always be positive for this to be a valid comparison
        print("Img:", i, "| PSNR Diff:", round(image[1] - image[2], 3), "| Error thresh:",
              image[3], " | Size diff:", image[4] - image[5], "bytes", "| Size ratio:", round(image[4] / image[5], 3))

        if i >= stop:
            break
    total_jpeg_size = sum([x[5] for x in image_settings if x])
    total_compressed_size = sum([x[4] for x in image_settings if x])
    print("Total JPEG size:", total_jpeg_size, "bytes")
    print("Total compressed size:", total_compressed_size, "bytes")
    print("Percentage of JPEG Size:", round(total_compressed_size / total_jpeg_size, 3)*100, "%")






compress_and_compare()





