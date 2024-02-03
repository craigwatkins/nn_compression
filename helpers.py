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



def get_ssim(original_path, compressed_path):
    from skimage import io, color
    from skimage.metrics import structural_similarity as ssim

    # Load the uncompressed and compressed images
    uncompressed_image = io.imread(original_path)
    compressed_image = io.imread(compressed_path)

    # get the min and max of the uncompressed image
    min_val = np.min(uncompressed_image)
    max_val = np.max(uncompressed_image)

    # Convert the images to grayscale
    uncompressed_gray = color.rgb2gray(uncompressed_image)
    compressed_gray = color.rgb2gray(compressed_image)

    # Compute SSIM
    ssim_index = ssim(uncompressed_gray, compressed_gray, data_range=max_val - min_val)
    return ssim_index

""""
ssim_index = get_ssim("images/test/reference.png", "images/test/reference.jpeg")
print("ssim", ssim_index)
ssim_index = get_ssim("images/test/reference.png", "images/test/test.png")
print("ssim", ssim_index)
"""







