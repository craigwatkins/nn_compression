# Random Nearest Neighbor Image Compression (RNNIC)

## Introduction
This project is a proof of concept for a novel technique in image compression, Random Nearest Neighbor Image Compression (RNNIC). This image compression algorithm leverages a relatively small lookup table of random samples to meet or exceed JPEG compression rates for high quality images.

## Why it Works
From the earliest days of image compression, color palettes have been used to compress images by selecting a few representative colors specific to that image. RNNIC uses a palette of the _differences_ between neighboring pixels instead. This allows the palette to represent a wide array of colors of varying hues, intensities and saturations with a comparatively small amount of values.

Unlike the distributions of colors, which vary greatly from image to image depending upon the subject matter, the differences between pixels are often quite similar across images. Thus, a random sampling from a variety of images can give a very flexible palette. Not only is this palette of differences relatively small, but it can be shared across innumerable images - it does not have to be redefined for each one.


## How it Works
This implementation of RNNIC employs a KD tree for finding the nearest neighbor in the differences palette to the original pixel differences. It then uses Huffman compression on the index values from the palette.

In principle, any sort of nearest neighbor algorithm can be effective and methods could be switched out rather easily. Approximate methods such as [ANNOY](https://pypi.org/project/annoy/) or [FAISS](https://ai.meta.com/tools/faiss/) could also be employed to increase performance.

A greedy algorithm is used to match the largest string of pixels it can while staying under a given error threshold. Kernels are used to identify likely strings of pixels that will meet the error threshold in order to reduce the number of comparisons required.

The current implementation requires a significant amount of computations per pixel, but has a linear time complexity with regard to image size. Decompression is straightforward and fast, primarily consisting of a hash table lookup after Huffman decompression.

## Results
To evaluate the effectiveness of this proof of concept, the images from the [Kodak dataset](https://r0k.us/graphics/kodak/), originally used to evaluate PNG compression, were employed. It is important to note, that the lookup table pixel samples were derived from a completely different set - the [Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k). Also, the total size of the palette when stored in database format, is less than 400 kilobytes. For comparison, storing the image data from a single image from the Kodak dataset in a similar, uncompressed fashion would take over 9 megabytes. 

For each images in the Kodak set, the compression settings were adjusted to achieve the same Peak Signal to Noise Ratio (PSNR) as the 100% quality JPEG version of the image. The result was that every single image achieved a higher compression ratio vs. JPEG and the overall compressed size of the Kodak dataset was improved by 16%. 

In the current proof of concept implementation, RNNIC does not degrade as gracefully as JPEG, so it’s advantages only appear at the very highest quality settings.


## How to Use
The project includes two scripts:
- `demo.py`: Demonstrates both compression and decompression for a single image.
- `kodak_test.py`: Compresses images in the Kodak dataset and compares them to JPEG versions.


# Installation Guide

## Prerequisites
Before installing the project dependencies, ensure you have Python installed on your system. This project is compatible with Python 3.6 and above.

## Installing Dependencies
This project relies on several third-party libraries, including OpenCV, NumPy, SciPy, and Numba. You can install these dependencies using pip, Python's package installer.

### Step-by-Step Instructions

1. **Open your terminal or command prompt.**

2. **Ensure pip is up to date:**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install OpenCV:**
   ```bash
   pip install opencv-python~=4.8.1.78
   ```

4. **Install NumPy:**
   ```bash
   pip install numpy~=1.26.2
   ```

5. **Install SciPy:**
   ```bash
   pip install scipy~=1.7.3
   ```

6. **Install Numba:**
   ```bash
   pip install numba~=0.58.1
   ```

## Post-Installation
After successfully installing these packages, verify the installation by checking the versions of the installed packages:

```bash
python -c "import cv2; print(cv2.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import numba; print(numba.__version__)"
```

If the above commands return the expected versions without errors, you have successfully installed all the required dependencies.

## Troubleshooting
If you encounter any issues during installation, consider the following:
- Check if your Python version is compatible.
- Ensure your pip installer is up to date.
- Verify that you have sufficient permissions to install packages.
- If a specific package fails to install, try installing it separately.

For more help, refer to the documentation of the respective packages or reach out to the community forums.


---