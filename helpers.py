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







