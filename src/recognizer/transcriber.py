import inspect, os, sys
from pathlib import Path

import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from natsort import os_sorted
from numpy.lib.function_base import place
from tap import Tap
from tensorflow import keras

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.recognizer.model import RecognizerModel
from src.data_handler.hebrew import HebrewAlphabet as Hebrew


class ArgParser(Tap):
    input_dir: Path  # Directory with character images

    def configure(self):
        self.add_argument("input_dir")

class Transcriber():

    def __init__(self) -> None:
        self.recognizer = RecognizerModel()
        self.recognizer.load_model("model1")

    def load_images(self, input_dir) -> np.array:
        character_images = []
        image_paths = []
        for img_path in os_sorted(input_dir.glob("*.png")):
            image_paths.append(img_path)
            print(f"Processing {img_path.name}")
            img = cv.imread(str(img_path))
            img = cv.resize(img, (60,70))
            img = np.expand_dims(img, axis = 0)
            character_images.append(img)
        return image_paths, character_images


if __name__ == "__main__":
    args = ArgParser(
        description="Input directory with character images"
    ).parse_args()

    transcriber = Transcriber()
    paths, input = transcriber.load_images(args.input_dir)

    current_line, current_word = 0, 0
    file = open(f"{args.input_dir}_characters.docx", "w") # Check if this is full path or name

    for i in range(len(input)):
        file_name = paths[i].name
        fragment_line, fragment_word, current_char = file_name.split("_")
        fragment_line = int(fragment_line.replace("characterL", ""))
        fragment_word = int(fragment_word.replace("W", ""))
        pred = np.argmax(transcriber.recognizer.predict(input[i]), axis=1)
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
