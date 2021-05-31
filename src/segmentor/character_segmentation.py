from pathlib import Path

import os
import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from glob import glob

class Character_segmentor():
    
    load_dotenv()

    def __init__(self) -> None:
        pass


    def place_holder(self):
        img_folder = "lines/"
        read_path = Path(os.environ["DATA_PATH"]) / img_folder
        cnt = 0
        for image_path in glob(f"{read_path}/*.png"):
            image = cv.imread(image_path)
            print(image_path)
            print(image.shape)
            tmp = image
            step_size = 2
            for width in [50,80,100]:
                window_size = (image.shape[0], width)
                for x in range(0, image.shape[1] - window_size[1], step_size):
                    window = image[x:x + window_size[1], 0: window_size[0], :]
                    #TODO: Feed window to CNN and see if its a letter and which
                break

if __name__ == "__main__":
    char_segmentor = Character_segmentor()
    char_segmentor.place_holder()