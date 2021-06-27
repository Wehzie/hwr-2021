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

    def __init__(self, model_name: str = "model_0") -> None:
        """Initialize the transcriber."""
        self.recognizer = RecognizerModel()
        self.recognizer.load_model(model_name)

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

    def transcribe_frag(self, input_dir: Path, output_dir: Path) -> None:
        """
        Transcribe a single fragment's characters.

        input_dir: Path to a single directory with extracted characters.
            For example data/fragment/characters/
            Where character contains instances of character_L*_W*_C*.png
        output_dir: Path to a directory to save the transcribed file.
        returns: None. Saves to file.
        """
        paths, input = self.load_images(input_dir)
        output_file = Path(f"{output_dir}/{input_dir.parent.name}_characters.txt")

        file = open(
            output_file,
            "w",
            encoding="utf8",
        )

        current_line, current_word = 0, 0
        for i in range(len(input)):
            file_name = paths[i].name
            _, fragment_line, fragment_word, _ = file_name.split("_")
            fragment_line = int(fragment_line.replace("L", ""))
            fragment_word = int(fragment_word.replace("W", ""))
            pred = np.argmax(self.recognizer.predict(input[i]), axis=1)[0]
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

    def match_character_folders(self, input_dir: Path) -> list:
        """
        Find folders named "characters" under the provided path.

        input_dir: Search under this path.
        return: List of Paths matching the query.
        """
        # segmentation saves characters under fragment/characters
        return input_dir.glob("**/characters")

    def transcribe_fragments(self, input_dir: Path, output_dir: Path) -> None:
        """ """
        output_dir.mkdir(parents=True, exist_ok=True)
        char_dirs: list = self.match_character_folders(input_dir)
        for char_dir in char_dirs:
            self.transcribe_frag(char_dir, output_dir)


if __name__ == "__main__":
    args = ArgParser(description="Input directory with character images").parse_args()
    transcriber = Transcriber()
    transcriber.transcribe_fragments(args.input_dir, args.output_dir)
