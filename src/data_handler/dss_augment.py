import cv2 as cv
import numpy as np
import os
import sys

from dotenv import load_dotenv
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.hebrew import HebrewAlphabet
from src.data_handler.imagemorph.imagemorph import elastic_morphing


class Augmenter:
    """Dead Sea Scrolls data augmenter."""

    load_dotenv()

    def __init__(self) -> None:
        """Initialize the augmenter."""
        pass

    def erosion_image(self, img_path, min_kernel_size, max_kernel_size) -> None:
        """Erode the image at the given path."""
        original = cv.imread(img_path)
        for i in range(min_kernel_size, max_kernel_size + 1):
            kernel = np.ones((i, i), np.uint8)
            eroded_img = cv.erode(original, kernel, iterations=1)
            cv.imwrite(f"{img_path}e{i}.jpg", eroded_img)

    def dilate_image(self, img_path, min_kernel_size, max_kernel_size) -> None:
        """Dilate the image at the given path."""
        original = cv.imread(img_path)
        for i in range(min_kernel_size, max_kernel_size + 1):
            kernel = np.ones((i, i), np.uint8)
            dilated_img = cv.dilate(original, kernel, iterations=1)
            cv.imwrite(f"{img_path}d{i}.jpg", dilated_img)

    def elastic_morphs(self, dir_path, reps):
        """Repeatedly apply morphing to character images of a font."""
        img_paths = glob(f"{dir_path}/*.jpg")
        amp = 2.5  # the amplitude of the deformation
        sigma = 10  # the local image area affected (spread of the gaussian smoothing kernel)
        for img_path in img_paths:
            print(img_path)
            img = cv.imread(str(img_path))
            h, w, _ = img.shape  # image height and width
            try:
                for rep in range(reps):
                    res = elastic_morphing(img, amp, sigma, h, w)  # morph image
                    cv.imwrite(f"{img_path}{rep}.jpg", res)  # write result to disk
            except:
                raise


if __name__ == "__main__":
    augment_tool = Augmenter()
    train_images = Path(os.environ["DATA_PATH"]) / "characters" / "train"
    hebrew = HebrewAlphabet()
    for letter in hebrew.letter_li:
        for i in (train_images / letter).glob("**/*.jpg"):
            augment_tool.dilate_image(i, 5, 5)
