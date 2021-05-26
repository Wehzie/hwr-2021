import sys
import inspect
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.data_handler.hebrew_alphabet import HebrewAlphabet

class FontImages:
    """
    Creates a folder with example images for each hebrew letter using habbakuk font.
    """

    load_dotenv()
    font_file = Path(os.environ["FONT_DATA"]) / "Habbakuk.TTF"
    training_folder = Path(os.environ["FONT_DATA"]) / "training"
    hebrew = HebrewAlphabet()

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
                Path(self.training_folder / Path(self.hebrew.letter_li[i] + ".jpeg")), "JPEG"
            )

    def assert_data_correct(self):
        if not Path.exists(self.training_folder):
            return False
        if len(os.listdir(self.training_folder)) != 27:
            return False
        return True


if __name__ == "__main__":
    font_img = FontImages()
    font_img.create_images()
