import argparse
from rnnic import NNCompressor  # Import your NNCompressor class
from helpers import save_image


def compress(args):
    compressor = NNCompressor()
    print("Compressing image...")
    compressor.compress(args.image_path, args.save_path, args.error_threshold, args.search_depth)
    print(f"Image compressed and saved as binary to {args.save_path}")


def decompress(args):
    decompressor = NNCompressor()
    decompressor.decompress(args.compressed_image_path)
    save_image(decompressor.decompressed_values, args.output_path)
    print(f"Image decompressed and saved as lossless PNG at {args.output_path}")


def main():
    # Initialize the main parser
    parser = argparse.ArgumentParser(description="NNCompressor CLI")
    subparsers = parser.add_subparsers(help='commands')

    # Subparser for the compress command
    compress_parser = subparsers.add_parser('compress', help='Compress an image')
    compress_parser.add_argument("image_path", help="Path to the image to compress")
    compress_parser.add_argument("save_path", help="Path to save the compressed image")
    compress_parser.add_argument("error_threshold", type=float, help="Error threshold for compression")
    compress_parser.add_argument("search_depth", type=int, help="Search depth for compression")
    compress_parser.set_defaults(func=compress)

    # Subparser for the decompress command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress an image')
    decompress_parser.add_argument("compressed_image_path", help="Path to the compressed image")
    decompress_parser.add_argument("output_path", help="Path to save the decompressed image")
    decompress_parser.set_defaults(func=decompress)

    # Parse arguments
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
