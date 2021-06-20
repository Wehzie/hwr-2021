from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from tap import Tap
from typing import List

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.segmentor.word_from_line import extract_words
from src.segmentor.line_extractor import extract_lines


class ArgParser(Tap):
    """Argument parser for character segmenter."""

    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted characters are saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


def middle_split(char: np.ndarray) -> List[np.ndarray]:
    middle = char.shape[1] // 2
    return [char[:, :middle], char[:, middle:]]


def split_connected(chars: List[np.ndarray]) -> List[np.ndarray]: 
    split_chars = []
    for char in chars:
        pixels = np.where(char == 0)
        height = pixels[0].max() - pixels[0].min()
        width = char.shape[1]
        # if width > 1.2 * height:
        #     split_chars.extend(middle_split(char))
        # else:
        split_chars.append(char)
        #print(f" max {height[0].max() - height[0].min()} total height: {char.shape[0]}")
    return split_chars


def extract_characters(char: np.ndarray, read_ord="r2l") -> List[np.ndarray]:
    """Load a word image and return extracted characters."""

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    characters = split_connected(img)

        # cv.imwrite(
        #    str((out_dir / f"{path_stem}c{i}.png").resolve()),
        #    cv.bitwise_not(clean),
        # )
    return characters


if __name__ == "__main__":
    args = ArgParser(
        description="Extract characters from binarized dead sea scroll pictures"
    ).parse_args()



    ### CHANGE: load only split images 


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
