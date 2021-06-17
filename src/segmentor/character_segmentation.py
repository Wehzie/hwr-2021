from pathlib import Path

import cv2 as cv
import numpy as np
import os
from skimage.morphology import skeletonize
from tap import Tap
from typing import List

from bounding_box import BoundingBox
from word_from_line import extract_words
from line_extractor import extract_lines


class ArgParser(Tap):
    """Argument parser for character segmenter."""

    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted characters are saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


def get_bounding_boxes(img: np.ndarray, min_pixel=120) -> List[BoundingBox]:
    """Get bounding boxes of characters in a word image.

    img: word image
    min_pixel: minimum amount of ink in a word
    L: filter length
    return: bounding boxes for each word in a line (from left to right)
    """
    img = img.copy()  # deep copy, not a reference
    new_img = img.copy()

    for i in range(len(new_img)):
        for j in range(len(new_img[i])):
            new_img[i][j] /= 255

    new_img = skeletonize(new_img)

    new_img = new_img.astype("uint8") * 255

    proj = np.sum(new_img, 0)
    proj = np.true_divide(proj, np.amax(proj))

    # idea: a word is a series of non-zeroes
    boxes = []
    height, width = img.shape
    box = BoundingBox(None, 0, None, height)
    for i, val in enumerate(proj):
        # character beginning
        if val != 0.0 and box.x is None:
            box.x = i  # set box x
        # character end
        if val == 0.0 and box.x is not None:
            box.w = i - box.x
            boxed_img = box.image_slice(img)
            # minimum number of non-white pixes
            if np.count_nonzero(boxed_img) > min_pixel:
                boxes.append(box)  # add box to list
            box = BoundingBox(None, 0, None, height)  # reset box
    return boxes

def split_non_connected(chars: List[np.ndarray]) -> List[np.ndarray]:
    return chars

def split_connected(chars: List[np.ndarray]) -> List[np.ndarray]:
    print(len(chars))
    # if len(chars):
    #     char = chars[0]
    #     height = np.where(char == 255)
    #     print(f" max {height[0].max()} min {height[0].min()} total height: {char.shape[0]}")
    #     return chars


def extract_characters(img: np.ndarray, read_ord="r2l") -> List[np.ndarray]:
    """Load a word image and return extracted characters."""

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)

    if read_ord == "r2l":  # words in order right to left
        boxes = reversed(boxes)

    characters: List[np.ndarray] = []
    for i, box in enumerate(boxes):
        # clean = clean_boxes(img, box)
        clean = box.image_slice(img)
        inversed = cv.bitwise_not(clean)
        if box.w > 16:
            characters.append(inversed)

        characters = split_non_connected(characters)
        characters = split_connected(characters)

        # cv.imwrite(
        #    str((out_dir / f"{path_stem}c{i}.png").resolve()),
        #    cv.bitwise_not(clean),
        # )
    return characters


if __name__ == "__main__":
    args = ArgParser(
        description="Extract characters from binarized dead sea scroll pictures"
    ).parse_args()

    # Extract lines
    for img_path in args.input_dir.glob("*binarized.jpg"):  # For each fragment
        print(f"Processing {img_path.name}")

        fragment_path = args.output_dir / img_path.stem
        character_path = fragment_path / "characters"
        character_path.mkdir(parents=True, exist_ok=True)

        image = cv.imread(str(img_path.resolve()))

        cv.imwrite(str((fragment_path / img_path.name).resolve()), image)
        lines = extract_lines(img_path)

        for i in range(len(lines)):  # Extract words
            line = lines[i]
            cv.imwrite(str((fragment_path / f"line{i}.png").resolve()), line)
            words = extract_words(line)

            for j in range(len(words)):  # Extract characters
                word = words[j]
                chars = extract_characters(word)
                for z in range(len(chars)):
                    char = chars[z]
                    # print(np.sum(char), i,j,z, char.shape)
                    cv.imwrite(str((character_path / f"characterL{i}_W{j}_C{z}.png").resolve()), char)
