import inspect, os, sys
from pathlib import Path

import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from numpy.lib.function_base import place
from tap import Tap
from tensorflow import keras

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.recognizer.model import RecognizerModel

class ArgParser(Tap):
    input_dir: Path  # Directory with character images

    def configure(self):
        self.add_argument("input_dir")

class seg_tester():

    def __init__(self) -> None:
        self.recognizer = RecognizerModel()
        self.recognizer.load_model("model1")
        

    def load_images(self, input_dir) -> np.array:
        character_images = []
        image_paths = []
        for img_path in input_dir.glob("*.png"):
            image_paths.append(img_path)
            print(f"Processing {img_path.name}")
            img = cv.imread(str(img_path))
            img = cv.resize(img, (60,70))
            img = np.expand_dims(img, axis = 0)
            #print(img.shape)
            character_images.append(img)
        return image_paths, character_images


if __name__ == "__main__":
    args = ArgParser(
        description="Input directory with character images"
    ).parse_args()

    testing = seg_tester()
    paths, input = testing.load_images(args.input_dir)
    
    for i in range(len(input)):
        name = paths[i]
        output = testing.recognizer.predict(input[i])
        print(name, np.argmax(output, axis=1))
        if output[0][np.argmax(output, axis=1)[0]] < 0.4:
            print(output)