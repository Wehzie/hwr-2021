from dataclasses import dataclass
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize
from tap import Tap
from typing import List

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.segmentor.bounding_box import BoundingBox
from src.segmentor.word_from_line import extract_words
from src.segmentor.line_extractor import extract_lines

MIN_WIDTH = 16
MAX_RATIO = 0.725


class ArgParser(Tap):
    """Argument parser for character segmenter."""

    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted characters are saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


@dataclass
class WriteParams:
    """Control what types of data are saved to file."""

    frag: bool = True
    line: bool = True
    word: bool = True
    char: bool = True


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


def middle_split(char: np.ndarray) -> List[np.ndarray]:
    middle = char.shape[1] // 2
    return [char[:, :middle], char[:, middle:]]


def split_connected(
    chars: List[np.ndarray], min_width: int, line_height: int
) -> List[np.ndarray]:
    split_chars = []
    for char in chars:
        pixels = np.where(char == 0)
        height = pixels[0].max() - pixels[0].min()
        width = char.shape[1]
        if width > 1.2 * height and width > 2 * min_width:
            split_chars.extend(middle_split(char))
        else:
            split_chars.append(char)
    return split_chars


def get_conn_components(img: np.ndarray, max_height=250, min_area=10000):

    img = img.copy()

    # Estimate connected components
    (num_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        img, 8, cv.CV_32S
    )

    imgs = []
    comp_ids = np.argsort(stats[:, cv.CC_STAT_LEFT])
    comp_ids = np.delete(comp_ids, np.where(comp_ids == 0))

    for comp_id in comp_ids:
        component_mask = (labels == comp_id).astype("uint8") * 255
        x = stats[comp_id, cv.CC_STAT_LEFT]
        y = stats[comp_id, cv.CC_STAT_TOP]
        w = stats[comp_id, cv.CC_STAT_WIDTH]
        h = stats[comp_id, cv.CC_STAT_HEIGHT]
        sub_mask = BoundingBox(x, y, w, h).image_slice(component_mask)
        blank = np.zeros((h, w), dtype=np.uint8)
        masked = np.where(sub_mask, 255, blank)
        imgs.append(masked)

    return imgs


def extract_characters_from_word(
    img: np.ndarray,
    line_height: int,
    read_ord="r2l",
    method="cca",
) -> List[np.ndarray]:
    """Load a word image and return extracted characters using vertical whitespace or CCA."""

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    if method == "cca":
        imgs = get_conn_components(img)
    else:
        boxes = get_bounding_boxes(img)
        imgs = [box.image_slice(img) for box in boxes]

    characters: List[np.ndarray] = []
    for i, char_img in enumerate(imgs):
        inversed = cv.bitwise_not(char_img)
        if char_img.shape[1] > MIN_WIDTH:
            characters.append(inversed)

    characters = split_connected(characters, MIN_WIDTH, line_height)

    if read_ord == "r2l":  # words in order right to left
        characters = reversed(characters)

    return characters


def estimate_line_height(
    img: np.ndarray,
):
    img = img.copy()

    # Estimate connected components
    (num_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        img, 8, cv.CV_32S
    )

    heights = []
    for h in range(1, num_labels):
        height = stats[h, cv.CC_STAT_HEIGHT]
        width = stats[h, cv.CC_STAT_WIDTH]
        area = stats[h, cv.CC_STAT_WIDTH] * stats[h, cv.CC_STAT_HEIGHT]
        pix_count = stats[h, cv.CC_STAT_AREA]
        ratio = pix_count / area

        if width > MIN_WIDTH:  # and ratio < MAX_RATIO:
            heights.append(height)

    return int(np.mean(heights)) if len(heights) > 0 else 0


def extract_chars_from_fragment(in_frag_path, output_dir, w_par):
    """
    in_frag_path: the path of the fragment from which characters are extracted
    output_dir: path of the directory to which characters are extracted
    w_par: parameters for writing to file controlled by a data class
    return: None. Writes to file
    """
    # Extract lines
    print(f"Processing {in_frag_path.name}")

    # define output paths and make directories
    out_frag_path: Path = output_dir / in_frag_path.stem
    line_path = out_frag_path / "lines"
    line_path.mkdir(parents=True, exist_ok=True)
    word_path = out_frag_path / "words"
    word_path.mkdir(parents=True, exist_ok=True)
    character_path = out_frag_path / "characters"
    character_path.mkdir(parents=True, exist_ok=True)

    # copy fragment image from input to output directory
    if w_par.frag == True:
        frag_img = cv.imread(str(in_frag_path.resolve()))
        cv.imwrite(str((out_frag_path / in_frag_path.name).resolve()), frag_img)
    # extract lines from a fragment
    lines = extract_lines(in_frag_path)

    for i, line in enumerate(lines):  # Extract words from lines
        # write line images to output directory
        if w_par.line == True:
            cv.imwrite(str((line_path / f"line_L{i}.png").resolve()), line)
        # extract words from a line
        words = extract_words(line)
        avg_line_height = estimate_line_height(line)

        for j, word in enumerate(words):  # Extract characters from words
            # write word images to output directory
            if w_par.word == True:
                cv.imwrite(str((word_path / f"word_L{i}_W{j}.png").resolve()), word)
            # extract characters from a word
            chars = extract_characters_from_word(word, line_height=avg_line_height)

            for z, char in enumerate(chars):
                # write character images to output directory
                if w_par.char == True:
                    char_path = character_path / f"character_L{i}_W{j}_C{z}.png"
                    area = char.shape[0] * char.shape[1]
                    pix_count = area - (np.sum(char) / 255)

                    if pix_count / area < MAX_RATIO:
                        cv.imwrite(str(char_path.resolve()), char)


if __name__ == "__main__":
    args = ArgParser(
        description="Extract characters from binarized dead sea scroll pictures."
    ).parse_args()

    write_params = WriteParams()

    # extract characters for each fragment
    for in_frag_path in args.input_dir.glob("*.jpg"):
        extract_chars_from_fragment(in_frag_path, args.output_dir, write_params)
