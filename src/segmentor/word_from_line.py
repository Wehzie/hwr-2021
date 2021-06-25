from pathlib import Path
import sys
from typing import List

import cv2 as cv
import numpy as np
from scipy.signal import lfilter
from tap import Tap

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.segmentor.bounding_box import BoundingBox


class ArgParser(Tap):
    """Argument parser for the words from lines extractor."""

    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted pictures are saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


def preprocess_img(img: np.ndarray, filter_len: int) -> np.ndarray:
    """Preprocess the image.

    img: line image
    filter_len: filter length
    return: collapsed, normalized and L-point filtered vector representation
    """
    # collapse matrix to vector
    # dim: x=line_width, y=1
    proj = np.sum(img, axis=0)  # sum over x-axis

    # normalization
    proj = np.true_divide(proj, np.amax(proj))

    # filtering/convolution
    # L is the filter length for an L-point averager
    b = (np.ones(filter_len)) / filter_len  # numerator coefficients
    a = np.ones(1)  # denominator coefficients
    # a similar result can be achieved with signal.convolve(proj, b)
    proj = lfilter(b, a, proj)  # filter output using lfilter function
    return proj


def get_bounding_boxes(
    img: np.ndarray,
    min_pixel=100,
    filter_len=10,
    min_width=16,
) -> List[BoundingBox]:
    """Get bounding boxes of the words in a line.

    img: line image
    min_pixel: minimum amount of ink in a word
    filter_len: filter length
    return: bounding boxes for each word in a line (from left to right)
    """
    img = img.copy()  # deep copy, not a reference
    proj = preprocess_img(img, filter_len)

    # idea: a word is a series of non-zeroes
    boxes = []
    height = img.shape[0]
    box = BoundingBox(None, 0, None, height)
    for i, val in enumerate(proj):
        # word beginning
        if val != 0.0 and box.x is None:
            box.x = i  # set box x
        # word end
        if val == 0.0 and box.x is not None:
            # -L because filter shifts peaks to the right
            box.w = i - box.x - filter_len
            boxed_img = box.image_slice(img)
            # minimum number of non-white pixels
            if np.count_nonzero(boxed_img) > min_pixel and box.w > min_width:
                boxes.append(box)  # add box to list
            box = BoundingBox(None, 0, None, height)  # reset box
    return boxes


def extract_words(img, read_ord="r2l") -> List[np.ndarray]:
    """Extract word images from a text line image.

    img_path: Path to a line image.
    read_ord: reading order (direction). right to left (r2l) or left to right (l2r)
    return: A list of extracted word images.
    """
    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)  # words in order left to right
    if read_ord == "r2l":  # words in order right to left
        boxes = reversed(boxes)

    return [cv.bitwise_not(box.image_slice(img)) for box in boxes]


def write_to_file(w_images: list, l_img_name: str, out_dir: Path) -> None:
    """Write a list of word images to a directory.

    w_images: A list of word images.
    l_img_name: The stem of the fiename of the line from which words are extracted.
    out_dir: The directory to which word images are written.
    return: None. Writes to file.
    """
    for idx, img in enumerate(w_images):
        out_path = Path(out_dir / (l_img_name + f"_W{idx}.png"))
        cv.imwrite(str(out_path.resolve()), cv.bitwise_not(img))


if __name__ == "__main__":
    args = ArgParser(
        description="Extract words from lines of binarized DSS images."
    ).parse_args()

    for l_img_path in args.input_dir.glob("*binarized_L*"):
        print(f"Processing {l_img_path.name}")
        w_images = extract_words(l_img_path)
        write_to_file(w_images, l_img_path.stem, args.output_dir)
