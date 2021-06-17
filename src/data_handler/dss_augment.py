import cv2 as cv
import numpy as np
import os
import sys

from dotenv import load_dotenv
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.hebrew import HebrewAlphabet


class Augmenter:
    """Dead Sea Scrolls data augmenter."""

    load_dotenv()

    def __init__(self) -> None:
        """Initialize the augmenter."""
        pass

    def erosion_image(self, img_path, min_kernel_size, max_kernel_size) -> None:
        """Erode the image at the given path."""
        original = cv.imread(img_path)
        for i in range(min_kernel_size, max_kernel_size):
            kernel = np.ones((i, i), np.uint8)
            eroded_img = cv.erode(original, kernel, iterations=1)
            cv.imwrite(f"{img_path}{i}.jpg", eroded_img)

    def dilate_image(self, img_path, min_kernel_size, max_kernel_size) -> None:
        """Dilate the image at the given path."""
        original = cv.imread(img_path)
        for i in range(min_kernel_size, max_kernel_size + 1):
            kernel = np.ones((i, i), np.uint8)
            dilated_img = cv.dilate(original, kernel, iterations=1)
            cv.imwrite(f"{img_path}{i}.jpg", dilated_img)


if __name__ == "__main__":
    augment_tool = Augmenter()
    train_images = Path(os.environ["DATA_PATH"]) / "characters" / "train"
    hebrew = HebrewAlphabet()
    for letter in hebrew.letter_li:
        for i in (train_images / letter).glob("**/*.jpg"):
            augment_tool.dilate_image(i, 5, 5)
