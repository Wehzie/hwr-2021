from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.signal import lfilter
from tap import Tap

from character_segmentation import extract_characters

class ArgParser(Tap):
    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted pictures are saved

    def configure(self):
        self.add_argument("input_dir")
        self.add_argument("output_dir")


class BoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
    
    def __str__(self) -> str:
        return f"x:{self.x}, y:{self.y}, w:{self.w}, h:{self.h}"

def preprocess_img(img: np.ndarray, L) -> np.ndarray:
    """
    img: line image
    L: filter length
    return: collapsed, normalized and L-point filtered vector representation
    """
    # collapse matrix to vector
    # dim: x=line_width, y=1
    proj = np.sum(img, 0)   # 0 = x-axis

    # normalization
    proj = np.true_divide(proj, np.amax(proj))

    # filtering/convolution
    # L is the filter length for an L-point averager
    b = (np.ones(L)) / L # numerator coefficients
    a = np.ones(1)  # denominator coefficients
    # a similar result can be achieved with signal.convolve(proj, b)
    proj = lfilter(b, a, proj) #filter output using lfilter function
    return proj


def get_bounding_boxes(img: np.ndarray, min_pixel=100, L=10) -> list:
    """
    img: line image
    min_pixel: minimum amount of ink in a word
    L: filter length
    return: bounding boxes for each word in a line (from left to right)
    """
    img = img.copy()    # deep copy, not a reference
    proj = preprocess_img(img, L)

    # idea: a word is a series of non-zeroes
    boxes = []
    height, width = img.shape 
    box = BoundingBox(None, 0, None, height)
    for i, val in enumerate(proj):
        # word beginning
        if val != 0.0 and box.x == None:
            box.x = i                                   # set box x
        # word end
        if val == 0.0 and box.x != None:
            # -L because filter shifts peaks to the right
            box.w = i-box.x-L
            boxed_img = img[box.y : box.y + box.h, box.x : box.x + box.w]
            # minimum number of non-white pixes
            if np.count_nonzero(boxed_img) > min_pixel:
                boxes.append(box)                           # add box to list
            box = BoundingBox(None, 0, None, height)    # reset box
    return boxes


def extract_words(img_path: Path, out_dir: Path) -> None:
    """
    Loads a line image and saves extracted words.
    """
    img = cv.imread(str(img_path.resolve()), cv.IMREAD_GRAYSCALE)
    assert np.all(img) != None, f"Image {img_path.name} not found!"

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)

    for i, box in enumerate(boxes):
        #clean = clean_boxes(img, box)
        clean = img[box.y : box.y + box.h, box.x : box.x + box.w]

        cv.imwrite(
            str((out_dir / f"{img_path.stem}_W{i}.png").resolve()),
            cv.bitwise_not(clean),
        )

        extract_characters(cv.bitwise_not(clean), Path(out_dir / "characters"), f"{img_path.stem}_W{i}")

if __name__ == "__main__":
    args = ArgParser(
        description="Extract words from lines of binarized DSS images."
    ).parse_args()

    # TODO: let loop iterates over all lines of all fragments
    for img in args.input_dir.glob("*binarized_L*"):
        print(f"Processing {img.name}")
        extract_words(img, args.output_dir)

