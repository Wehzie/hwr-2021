import sys
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from natsort import os_sorted
from tap import Tap

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.recognizer.model import RecognizerModel
from src.data_handler.hebrew import HebrewAlphabet as Hebrew


class ArgParser(Tap):
    """Argument parser for the transcriber."""

    input_dir: Path  # Directory with character images
    output_dir: Path  # Directory to which txt is saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


class Transcriber:
    """Transcriber from extracted characters to a text file."""

    def __init__(self) -> None:
        """Initialize the transcriber."""
        self.recognizer = RecognizerModel()
        self.recognizer.load_model("model1")

    def load_images(self, input_dir: Path) -> Tuple[List[Path], List[np.ndarray]]:
        """Load images from the input directory."""
        character_images = []
        image_paths = []
        for img_path in os_sorted(input_dir.glob("*.png")):
            image_paths.append(img_path)
            img = cv.imread(str(img_path))
            img = cv.resize(img, (60, 70))
            img = np.expand_dims(img, axis=0)
            character_images.append(img)
        return image_paths, character_images


if __name__ == "__main__":
    args = ArgParser(description="Input directory with character images").parse_args()

    transcriber = Transcriber()
    paths, input = transcriber.load_images(Path(args.input_dir))

    current_line, current_word = 0, 0
    file = open(
        f"{args.output_dir}/{args.input_dir.parent.name}_characters.txt",
        "w",
        encoding="utf8",
    )  # Check if this is full path or name

    for i in range(len(input)):
        file_name = paths[i].name
        _, fragment_line, fragment_word, current_char = file_name.split("_")
        fragment_line = int(fragment_line.replace("L", ""))
        fragment_word = int(fragment_word.replace("W", ""))
        pred = np.argmax(transcriber.recognizer.predict(input[i]), axis=1)[0]
        uni_char = Hebrew.unicode_dict[Hebrew.letter_li[pred]]
        if fragment_line > current_line:
            file.write(f"\n{uni_char}")
            current_line += 1
            current_word = 0
        elif fragment_word > current_word:
            file.write(f" {uni_char}")
            current_word += 1
        else:
            file.write(uni_char)
    file.close()
