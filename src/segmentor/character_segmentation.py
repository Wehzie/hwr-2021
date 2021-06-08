from pathlib import Path

import cv2 as cv
import numpy as np
import os
from scipy.signal import lfilter
from scipy import ndimage
from skimage.morphology import skeletonize
from tap import Tap

from word_from_line import extract_words
from line_extractor import extract_lines

class ArgParser(Tap):
    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted characters are saved

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

def get_bounding_boxes(img: np.ndarray, min_pixel=120) -> list:
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


def extract_characters(img: np.ndarray, read_ord = "r2l") -> None:
    """
    Loads a word image and returns extracted characters.
    """

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)

    #if read_ord == "r2l":   # words in order right to left
    #    boxes = reversed(boxes)

    characters = []
    for i, box in enumerate(boxes):
        #clean = clean_boxes(img, box)
        clean = img[box.y : box.y + box.h, box.x : box.x + box.w]
        characters.append(cv.bitwise_not(clean))
        #cv.imwrite(
        #    str((out_dir / f"{path_stem}c{i}.png").resolve()),
        #    cv.bitwise_not(clean),
        #)
    return characters

if __name__ == "__main__":
    args = ArgParser(
        description="Extract characters from binarized dead sea scroll pictures"
    ).parse_args()

    # Extract lines
    for img in args.input_dir.glob("*binarized.jpg"): # For each fragment
        print(f"Processing {img.name}")
        
        img_path = str((args.input_dir / f"{img.name}").resolve())
        fragment_path =  str((args.output_dir / f"{img.name}").resolve())[:-4]
        character_path = fragment_path + "/" + "characters"
        os.mkdir(fragment_path)
        os.mkdir(character_path)

        image = cv.imread(img_path)

        cv.imwrite(fragment_path + "/" + f"{img.name}", image)
        lines = extract_lines(img)
        
        for i in range(len(lines)): # Extract words
            line = lines[i]
            cv.imwrite(f"{fragment_path}/line{i}.png", line)
            words = extract_words(line)

            for j in range(len(words)): # Extract characters
                word = words[j]
                chars = extract_characters(word)
                for z in range(len(chars)):
                    char = chars[z]
                    cv.imwrite(f"{character_path}/characterL{i}_W{j}_C{z}.png", char)

