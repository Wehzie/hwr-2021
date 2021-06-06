from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.signal import lfilter
from scipy import ndimage
from skimage.morphology import skeletonize
from tap import Tap

class BoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
    
    def __str__(self) -> str:
        return f"x:{self.x}, y:{self.y}, w:{self.w}, h:{self.h}"

def get_bounding_boxes(img: np.ndarray, min_pixel=100) -> list:
    """
    img: word image
    min_pixel: minimum amount of ink in a word
    L: filter length
    return: bounding boxes for each word in a line (from left to right)
    """
    img = img.copy()    # deep copy, not a reference
    new_img = img.copy()
    
    for i in range(len(new_img)):
        for j in range(len(new_img[i])):
            new_img[i][j] /= 255

    new_img = skeletonize(new_img)

    new_img = new_img.astype('uint8') * 255

    proj = np.sum(new_img, 0)
    proj = np.true_divide(proj, np.amax(proj))

    # idea: a word is a series of non-zeroes
    boxes = []
    height, width = img.shape 
    box = BoundingBox(None, 0, None, height)
    for i, val in enumerate(proj):
        # character beginning
        if val != 0.0 and box.x == None:
            box.x = i                                   # set box x
        # character end
        if val == 0.0 and box.x != None:
            box.w = i-box.x
            boxed_img = img[box.y : box.y + box.h, box.x : box.x + box.w]
            # minimum number of non-white pixes
            if np.count_nonzero(boxed_img) > min_pixel:
                boxes.append(box)                           # add box to list
            box = BoundingBox(None, 0, None, height)    # reset box
    return boxes


def extract_characters(img: np.ndarray, out_dir: Path, path_stem: Path) -> None:
    """
    Loads a word image and returns extracted characters.
    """

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)

    for i, box in enumerate(boxes):
        #clean = clean_boxes(img, box)
        clean = img[box.y : box.y + box.h, box.x : box.x + box.w]
        cv.imwrite(
            str((out_dir / f"{path_stem}c{i}.png").resolve()),
            cv.bitwise_not(clean),
        )


if __name__ == "__main__":
    input_dir = Path("data/words/")
    output_dir = Path("data/segmented_chars/")
    for img in input_dir.glob("*binarized_L*"):
        print(f"Processing {img.name}")
        extract_characters(img, output_dir)
