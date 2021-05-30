import inspect
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.data_handler.hebrew_alphabet import HebrewAlphabet
from src.data_handler.imagemorph.imagemorph import elastic_morphing

class FontImages:
    """
    Creates a folder with example images for each hebrew letter using habbakuk font.
    """

    load_dotenv()
    font_file = Path(os.environ["FONT_DATA"]) / "Habbakuk.TTF"
    training_folder = Path(os.environ["FONT_DATA"]) / "training"
    hebrew = HebrewAlphabet()
    
    #### ELASTIC MORPHING PARAMETERS
    amp = 0.9   # the amplitude of the deformation
    sigma = 9   # the local image area affected (spread of the gaussian smoothing kernel)
    repetitions = 30    # the number of morphed images produced for each character

    def __init__(self) -> None:
        pass

    def create_images(self) -> np.ndarray:
        font = ImageFont.truetype(str(self.font_file), 45, encoding="utf-8")
        if not Path.exists(self.training_folder):
            os.mkdir(self.training_folder)
        for i in range(len(self.hebrew.font_li)):
            text = self.hebrew.font_li[i]
            text_width, text_height = font.getsize(text)
            canvas = Image.new("RGB", (text_width + 15, text_height + 20), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), text, "black", font)
            canvas.save(
                Path(self.training_folder / Path(self.hebrew.letter_li[i] + ".jpeg")),
                "JPEG",
            )

    def assert_data_correct(self):
        if not Path.exists(self.training_folder):
            return False
        if len(os.listdir(self.training_folder)) != 27:
            return False
        return True

    def augment_data(self):
        """
        Repeatedly apply morphing to character images of a font.
        """
        for char in self.hebrew.letter_li:
            char_path = self.training_folder / char
            try:
                os.mkdir(char_path)   # make directory for each character
            except FileExistsError:
                print(f"The folder {char_path} already exists.")
            img = cv.imread(str(self.training_folder / char) + ".jpeg") # read font character
            h, w, _ = img.shape                     # image height and width

            for rep in range(self.repetitions):
                res = elastic_morphing(img, self.amp, self.sigma, h, w) # morph image
                write_path = str(char_path / char) + str(rep) + ".jpeg"
                cv.imwrite(write_path, res) # write result to disk

if __name__ == "__main__":
    font_img = FontImages()
    #font_img.create_images()
    font_img.augment_data()
