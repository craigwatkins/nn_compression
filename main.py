import cv2
from helpers import save_image, calculate_psnr, save_jpeg
from nNComressor import NNCompressor
import time


def make_reference_image(image_path, crop_size=100, crop_offset=0, file_path="images/reference", crop_image=True):
    # open image from filepath, crop it, and save as "reference.png"
    img = cv2.imread(image_path)
    if crop_image:
        img = img[crop_offset:crop_size, crop_offset:crop_size]
    save_image(img, file_path+".png")
    save_jpeg(img, file_path+".jpeg", 100)


def demo():
    crop_size = 450
    crop_offset = 150
    image = "19"
    source_path = f"""images/kodak/kodim{image}.png"""
    save_path = f"""bins/test/{image}.bin"""
    error_threshold = 5
    make_reference_image(source_path, crop_size, crop_offset)

    print("compressing image")
    compressed = NNCompressor()
    compressed.compress('images/test/reference.png', save_path, error_threshold)
    decompressor = NNCompressor()
    decompressor.decompress(save_path)
    # save the regenerated image
    save_image(compressed.compressed_values, "images/test/test.png")
    save_image(decompressor.decompressed_values, "images/test/verify.png")
    psnr_v = calculate_psnr("images/test/reference.png", "images/test/test.png")
    print("vector psnr", psnr_v)
    psnr_j = calculate_psnr("images/test/reference.png", "images/test/reference.jpeg")
    psnr_verify = calculate_psnr("images/test/test.png", "images/test/verify.png")
    print("jpeg psnr", psnr_j)
    print("verify psnr", psnr_verify)


if __name__ == "__main__":
    demo()

