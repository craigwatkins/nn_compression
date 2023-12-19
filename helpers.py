import numpy as np
import cv2


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







