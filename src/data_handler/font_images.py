import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


class fontImages:
    """
    Creates a folder with example images for each hebrew letter using habbakuk font.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.font_file = str(Path(os.environ["FONT_DATA"])) + "\\Habbakuk.ttf"
        self.training_folder = Path(os.environ["FONT_DATA"] + "training")
        self.alphabet = [
            ")",
            "(",
            "b",
            "d",
            "g",
            "h",
            "x",
            "k",
            "\\",
            "l",
            "m",
            "{",
            "n",
            "}",
            "p",
            "v",
            "q",
            "r",
            "s",
            "$",
            "t",
            "+",
            "j",
            "c",
            "w",
            "y",
            "z",
        ]
        self.letters = [
            "Alef",
            "Ayin",
            "Bet",
            "Dalet",
            "Gimel",
            "He",
            "Het",
            "Kaf",
            "Kaf-final",
            "Lamed",
            "Mem",
            "Mem-medial",
            "Nun-final",
            "Nun-medial",
            "Pe",
            "Pe-final",
            "Qof",
            "Resh",
            "Samekh",
            "Shin",
            "Taw",
            "Tet",
            "Tsadi-final",
            "Tsadi-medial",
            "Waw",
            "Yod",
            "Zayin",
        ]

    def create_images(self) -> np.ndarray:
        print(self.font_file)
        font = ImageFont.truetype(self.font_file, 45, encoding="utf-8")
        if not Path.exists(self.training_folder):
            os.mkdir(self.training_folder)
        for i in range(len(self.alphabet)):
            text = self.alphabet[i]
            text_width, text_height = font.getsize(text)
            canvas = Image.new("RGB", (text_width + 15, text_height + 20), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), text, "black", font)
            canvas.save(
                Path(self.training_folder / Path(self.letters[i] + ".jpeg")), "JPEG"
            )

    def assert_data_correct(self):
        if not Path.exists(self.training_folder):
            return False
        if len(os.listdir(self.training_folder)) != 27:
            return False
        return True


if __name__ == "__main__":
    font_img = fontImages()
    font_img.create_images()
