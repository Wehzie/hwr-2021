import cv2 as cv
import numpy as np
import inspect, os, sys

from dotenv import load_dotenv
from glob import glob
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
print(project_root_dir)
sys.path.insert(0, project_root_dir)

from src.data_handler.hebrew import HebrewAlphabet

class Augmenter():

    load_dotenv()
    
    def __init__(self) -> None:
        pass

    def erosion_image(self, img_path, min_kernel_size, max_kernel_size):
        original = cv.imread(img_path)
        for i in range(min_kernel_size, max_kernel_size):
            kernel = np.ones((i,i), np.uint8)
            eroded_img = cv.erode(original, kernel, iterations=1)
            cv.imwrite(f"{img_path}{i}.jpg")

    def dilate_image(self, img_path, min_kernel_size, max_kernel_size):
        original = cv.imread(img_path)
        for i in range(min_kernel_size, max_kernel_size+1):
            kernel = np.ones((i,i), np.uint8)
            dilated_img = cv.dilate(original, kernel, iterations=1)
            cv.imwrite(f"{img_path}{i}.jpg", dilated_img)

if __name__ == "__main__":
    augment_tool = Augmenter()
    train_images = Path(os.environ["DATA_PATH"]) / "characters" / "train"
    hebrew = HebrewAlphabet()
    for letter in hebrew.letter_li:
        images = glob(f'{Path(train_images/letter)}/*.jpg')
        for i in images:
            augment_tool.dilate_image(i, 5,5)