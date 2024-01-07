# Random Nearest Neighbor Compression (RNNC)

## Introduction
This project introduces a novel technique in image compression, named Random Nearest Neighbor Compression (RNNC). RNNC utilizes nearest neighbors from randomly sampled values to replace original values, effectively compressing images with fewer samples yet achieving or exceeding JPEG compression rates at 100% quality settings.

## Why it Works
RNNC differs from early image compression methods like GIFs by using a palette of pixel differences, allowing representation of a wide range of colors with a small set of values. The method leverages the similarity in pixel differences across images, and the associated nature of RGB color channels, to create an efficient and shared palette of differences.

## How it Works
RNNC employs a KD tree for finding the nearest neighbor in the differences palette and uses Huffman compression on the index values. The compression algorithm has a linear time complexity with image size, while decompression is straightforward, primarily involving hash table lookups after Huffman decompression.

## Results
Using the Kodak images for evaluation and the Flickr dataset for palette derivation, RNNC achieved higher compression ratios compared to 100% quality JPEG versions, with a 16% overall improvement on the Kodak set.

## How to Use
The project includes two scripts:
- `demo.py`: Demonstrates both compression and decompression for a single image.
- `kodak_test.py`: Compresses images in the Kodak dataset and compares them to JPEG versions.

## Installation and Setup
(Include steps for installation, setup, and any dependencies required.)

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

(Consider adding any additional sections like FAQs, Background and Motivation, Literature References, Testimonials, or Community and Support information.)

