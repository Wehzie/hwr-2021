import os
import sys
from pathlib import Path

import cv2 as cv
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.hebrew import HebrewAlphabet
from src.data_handler.imagemorph.imagemorph import elastic_morphing


class FontImages:
    """Creates a folder with example images for each hebrew letter using habbakuk font."""

    load_dotenv()

    font_file = Path(os.environ["FONT_DATA"]) / "Habbakuk.TTF"
    training_folder = Path(os.environ["FONT_DATA"]) / "training"
    hebrew = HebrewAlphabet()

    #### ELASTIC MORPHING PARAMETERS
    amp = 2.5  # the amplitude of the deformation
    sigma = (
        10  # the local image area affected (spread of the gaussian smoothing kernel)
    )
    repetitions = 30  # the number of morphed images produced for each character

    def create_images(self):
        """Write a .jpeg image for each character in the font character set."""
        font = ImageFont.truetype(str(self.font_file), 45, encoding="utf-8")
        for letter in self.hebrew.letter_li:
            (self.training_folder / letter).mkdir(parents=True, exists_ok=True)

        for i in range(len(self.hebrew.font_li)):
            letter_path = self.training_folder / Path(self.hebrew.letter_li[i])
            text = self.hebrew.font_li[i]
            text_width, text_height = font.getsize(text)
            canvas = Image.new("RGB", (text_width + 15, text_height + 20), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), text, "black", font)
            canvas.save(
                Path(letter_path / Path(f"{self.hebrew.letter_li[i]}_original.jpeg")),
                "JPEG",
            )

    def assert_data_correct(self) -> bool:
        """Assert that the font data exists and is in the correct format."""
        if not self.training_folder.exists():
            return False
        # 27: number of characters
        # 27*2: 27 original font characters and 27 folders with morphed version
        if len(list(self.training_folder.iterdir())) not in [27, 27 * 2]:
            return False
        # assert that each character folder has the expected number of images inside
        # expected number is repetitions + original, or just original if no morphing
        # took place
        for directory in self.training_folder.iterdir():
            img_count = len(list((self.training_folder / directory).iterdir()))
            if img_count != self.repetitions + 1 and img_count != 1:
                return False
        return True

    def augment_data(self):
        """Repeatedly apply morphing to character images of a font."""
        for char in self.hebrew.letter_li:
            char_path = self.training_folder / char
            img = cv.imread(
                str((self.training_folder / char / f"{char}_original.jpeg").resolve())
            )  # read font character
            h, w, _ = img.shape  # image height and width

            for rep in range(self.repetitions):
                res = elastic_morphing(img, self.amp, self.sigma, h, w)  # morph image
                cv.imwrite(char_path / f"{char}{rep}.jpeg", res)  # write result to disk


if __name__ == "__main__":
    font_img = FontImages()
    font_img.create_images()
    try:
        font_img.augment_data()
    except:
        pass
