import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from natsort import os_sorted
from tap import Tap

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.recognizer.model import RecognizerModel


class ArgParser(Tap):
    """Argument parser for segmenter testing."""

    input_dir: Path  # Directory with character images

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")


class SegTester:
    """Segmentation tester."""

    def __init__(self) -> None:
        """Initialize the segmentation tester."""
        self.recognizer = RecognizerModel()
        self.recognizer.load_model("model1")

    def load_images(self, input_dir) -> np.array:
        """Load images."""
        character_images = []
        image_paths = []
        for img_path in os_sorted(input_dir.glob("*.png")):
            image_paths.append(img_path)
            print(f"Processing {img_path.name}")
            img = cv.imread(str(img_path))
            img = cv.resize(img, (60, 70))
            img = np.expand_dims(img, axis=0)
            # print(img.shape)
            character_images.append(img)
        return image_paths, character_images


if __name__ == "__main__":
    args = ArgParser(description="Input directory with character images").parse_args()

    testing = SegTester()
    paths, input = testing.load_images(args.input_dir)

    for i in range(len(input)):
        name = paths[i]
        output = testing.recognizer.predict(input[i])
        print(name, np.argmax(output, axis=1))
        print(output, sorted(output[0], reverse=True)[:4])
    print(len(input))
