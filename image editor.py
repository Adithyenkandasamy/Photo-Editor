import argparse
from PIL import Image
import os


def process_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Perform some basic image processing (e.g., convert to grayscale)
        grayscale_img = img.convert("L")
        # Save the processed image
        base, ext = os.path.splitext(image_path)
        new_image_path = f"{base}_grayscale{ext}"
        grayscale_img.save(new_image_path)
        print(f"Processed image saved as {new_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Process an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    process_image(args.image_path)


if __name__ == "__main__":
    main()