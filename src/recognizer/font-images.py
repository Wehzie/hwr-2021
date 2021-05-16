import os	
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class fontImages():
    """
    
    """

    def __init__(self) -> None:
        load_dotenv()
        self.font_file = str(Path(os.environ['FONT_DATA']))
        self.alphabet = ['),', '(,', 'b,', 'd,', 'g,', 'h,', 'x,', 'k,',
                         '\\', 'l,', 'm,', '{,', 'n,', '},', 'p,', 'v,',
                         'q,', 'r,', 's,', '$,', 't,', '+,', 'j,', 'c,',
                         'w,', 'y,', 'z']

    def create_images(self) -> np.ndarray:
        print(self.font_file)
        font = ImageFont.truetype(self.font_file, 35, encoding="utf-8")
        cnt = 0
        for i in self.alphabet:
            if cnt == 3:
                break
            text_width, text_height = font.getsize(i)
            canvas = Image.new('RGB', (text_width + 20, text_height + 20), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), i, 'black', font)
            canvas.show()
            cnt += 1
            

if __name__ == "__main__":
    font_img = fontImages()
    font_img.create_images()