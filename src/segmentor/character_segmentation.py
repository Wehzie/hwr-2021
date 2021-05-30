from pathlib import Path

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from glob import glob

class CharacterSegmentor():
    
    load_dotenv()

    def __init__(self) -> None:
        pass


    def place_holder(self):
        img_folder = "lines/"
        read_path = Path(os.environ["DATA_PATH"]) / img_folder
        for image_path in glob(f"{read_path}/*.png"):
            image = cv.imread(image_path)
            print(image_path)
            print(image.shape)
            tmp = image
            step_size = 40
            window_size = (68,40)
            for x in range(0, image.shape[1] - window_size[1], step_size):
                window = image[x:x + window_size[1], 0: window_size[0], :]
                # draw rectangle on image
                cv.rectangle(tmp, (x, 0), (x + window_size[1], window_size[0]), (255, 0, 0), 2)
                plt.imshow(np.array(tmp).astype('uint8'))
            plt.savefig('boxes.png')
            break

if __name__ == "__main__":
    char_segmentor = CharacterSegmentor()
    char_segmentor.place_holder()