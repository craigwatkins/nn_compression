import cv2
from src.helpers import save_image, calculate_psnr, save_jpeg
from src import NNCompressor
import os

IMAGES_FOLDER = "images"
BIN_FOLDER = "bins"
def get_image_path(folder, file_name=""):
    # Path to the script being executed
    script_path = os.path.abspath(__file__)
    # Directory containing the script
    script_dir = os.path.dirname(script_path)
    # get the parent directory of the current directory
    parent_dir = os.path.dirname(script_dir)
    # add the image directory to the parent directory
    image_dir = os.path.join(parent_dir, folder, file_name)
    return image_dir


def make_reference_image(image_path, crop_size=100, crop_offset=0, file_path="../images/test/reference", crop_image=True):
    # open image from filepath, crop it, and save as "reference.png"
    img = cv2.imread(image_path)
    if crop_image:
        img = img[crop_offset:crop_size, crop_offset:crop_size]
    save_image(img, file_path+".png")
    save_jpeg(img, file_path+".jpeg", 100)


def decompress_image(compressed_path, test_image_path, verify_image_path):
    decompressor = NNCompressor()
    decompressor.decompress(compressed_path)
    save_image(decompressor.decompressed_values, verify_image_path)
    psnr_verify = calculate_psnr(test_image_path, verify_image_path)
    print("verify psnr", psnr_verify)


def demo():
    crop_size = 250
    crop_offset = 150
    crop_image = True
    image = "16"
    source_path = get_image_path(IMAGES_FOLDER, f"kodak/kodim{image}.png")
    save_path = get_image_path(BIN_FOLDER, f"test/{image}.bin")
    reference_image_path = get_image_path(IMAGES_FOLDER, f"test/reference.png")
    reference_jpeg_path = get_image_path(IMAGES_FOLDER, f"test/reference.jpeg")
    test_image_path = get_image_path(IMAGES_FOLDER, f"test/test.png")
    verify_image_path = get_image_path(IMAGES_FOLDER, f"test/verify.png")

    # error_threshold controls the quality of the compression - lowering error threshold increases quality
    error_threshold = 3.1
    # make a lossless reference image that can be used to check the compression quality
    make_reference_image(source_path, crop_size, crop_offset, file_path=reference_image_path, crop_image=crop_image)

    print("compressing image")
    compressed = NNCompressor()
    compressed.compress(reference_image_path, save_path, error_threshold)
    # save the regenerated image
    save_image(compressed.compressed_values, test_image_path)
    # Technically, there is no need to delete the compression object, as it is intended to be used multiple times.
    # This is just a precaution to ensure that there is no data leakage during development.
    del compressed
    decompress_image(save_path, test_image_path, verify_image_path)

    psnr_v = calculate_psnr(reference_image_path, test_image_path)
    print("vector psnr", psnr_v)
    # create a high quality jpeg image for comparison
    psnr_j = calculate_psnr(reference_image_path, reference_jpeg_path)
    print("jpeg psnr", psnr_j)


if __name__ == "__main__":
    demo()
