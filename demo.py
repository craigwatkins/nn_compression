import cv2
from helpers import save_image, calculate_psnr, save_jpeg
from rnnic import NNCompressor


def make_reference_image(image_path, crop_size=100, crop_offset=0, file_path="images/test/reference", crop_image=True):
    # open image from filepath, crop it, and save as "reference.png"
    img = cv2.imread(image_path)
    if crop_image:
        img = img[crop_offset:crop_size, crop_offset:crop_size]
    save_image(img, file_path+".png")
    save_jpeg(img, file_path+".jpeg", 100)


def decompress_image(save_path):
    decompressor = NNCompressor()
    decompressor.decompress(save_path)
    save_image(decompressor.decompressed_values, "images/test/verify.png")
    psnr_verify = calculate_psnr("images/test/test.png", "images/test/verify.png")
    print("verify psnr", psnr_verify)


def demo():
    crop_size = 250
    crop_offset = 150
    crop_image = False
    image = "16"
    source_path = f"""images/kodak/kodim{image}.png"""
    save_path = f"""bins/test/{image}.bin"""
    # error_threshold controls the quality of the compression - lowering error threshold increases quality
    error_threshold = 3000
    # search_depth controls the thoroughness of the search - higher values are slower and may improve compression ratio
    search_depth = 10
    # make a lossless reference image that can be used to check the compression quality
    make_reference_image(source_path, crop_size, crop_offset, crop_image=crop_image)

    print("compressing image")
    compressed = NNCompressor()
    compressed.compress('images/test/reference.png', save_path, error_threshold, search_depth)
    # save the regenerated image
    save_image(compressed.compressed_values, "images/test/test.png")
    # Technically, there is no need to delete the compression object, as it is intended to be used multiple times.
    # This is just a precaution to ensure that there is no data leakage during development.
    del compressed
    decompress_image(save_path)

    psnr_v = calculate_psnr("images/test/reference.png", "images/test/test.png")
    print("vector psnr", psnr_v)
    # create a high quality jpeg image for comparison
    psnr_j = calculate_psnr("images/test/reference.png", "images/test/reference.jpeg")
    print("jpeg psnr", psnr_j)

    # psnr_j = calculate_psnr("images/test/reference.png", "images/test/reference.webp")
    # print("webp psnr", psnr_j)
    # psnr_j = calculate_psnr("images/test/reference.png", "images/test/verify.png")



if __name__ == "__main__":
    demo()
