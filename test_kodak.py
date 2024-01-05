import cv2
from helpers import save_image, save_jpeg
from nNComressor import NNCompressor
import os
import numpy as np
import math


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

def compress_image(image_path, save_path, error_threshold, search_depth):
    compressor = NNCompressor()
    compressed = compressor.compress(image_path, save_path, error_threshold, search_depth)
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
    images = get_images("images/kodak")
    error_thresholds = [0.0]*25
    error_thresholds[0] = 3.8
    error_thresholds[1] = 6
    error_thresholds[2] = 4.6
    error_thresholds[3] = 5.1
    error_thresholds[4] = 6.1
    error_thresholds[5] = 4.3
    error_thresholds[6] = 4.7
    error_thresholds[7] = 6.3
    error_thresholds[8] = 3.8
    error_thresholds[9] = 4.1
    error_thresholds[10] = 4.1
    error_thresholds[11] = 3.8
    error_thresholds[12] = 5.5
    error_thresholds[13] = 6
    error_thresholds[14] = 6
    error_thresholds[15] = 0
    error_thresholds[16] = 3.4
    error_thresholds[17] = 5.6
    error_thresholds[18] = 4.0
    error_thresholds[19] = 5.6
    error_thresholds[20] = 4.4
    error_thresholds[21] = 5.4
    error_thresholds[22] = 4.2
    error_thresholds[23] = 6.5
    image_settings = [[] for i in range(len(images))]
    start = 0
    stop = 23
    for i, image in enumerate(images):
        if i < start:
            continue
        compressed = compress_image(image[0], f"bins/test/{image[1]}.bin", error_thresholds[i], 1000)
        # save compressed image
        psnr = calculate_psnr(image[0], compressed)
        save_jpeg(cv2.imread(image[0]), f"images/test/{image[1]}.jpeg", 100)
        psnr_jpeg = calculate_psnr(image[0], cv2.imread(f"images/test/{image[1]}.jpeg"))
        # get jpeg size
        jpeg_size = os.path.getsize(f"images/test/{image[1]}.jpeg")
        # get compressed size
        compressed_size = os.path.getsize(f"bins/test/{image[1]}.bin")
        image_settings[i] = (image[1], psnr, psnr_jpeg, error_thresholds[i], compressed_size, jpeg_size)
        image = image_settings[i]
        print("Img:", i, "| Quality (+):", round(image[1] - image[2], 3), "| Error thresh:",
              image[3], " | Size diff:", image[4] - image[5], "bytes", "| Size ratio:", round(image[4] / image[5], 3))

        if i >= stop:
            break





compress_and_compare()





