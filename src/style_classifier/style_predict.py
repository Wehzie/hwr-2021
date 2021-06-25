import os
import sys
import numpy as np
import cv2 as cv
from pathlib import Path
from tap import Tap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model import StyleClassifierModel
from style_train import CV_IMG_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_handler.hebrew import HebrewStyles


class ArgParser(Tap):
    """Argument parser for the style predictor."""

    input_dir: Path  # Directory that contains segmented characters
    output_dir: Path  # Directory where prediction will be saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


if __name__ == "__main__":
    args = ArgParser(description="Predict style from segmented characters").parse_args()
    assert args.input_dir.exists(), "Input directory does not exist"
    assert args.output_dir.exists(), "Output directory does not exist"

    imgs = []
    for file_path in args.input_dir.glob("*.png"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, CV_IMG_SIZE)
        imgs.append(img)
    imgs = np.array(imgs)

    style_model = StyleClassifierModel()
    style_model.load_model("style-classifier")

    preds_sum = np.sum(style_model.predict(imgs), axis=0)
    print(preds_sum / len(imgs))
    style_pred = np.argmax(preds_sum)
    style_string = HebrewStyles.style_li[style_pred]

    file_name = args.input_dir.parents[0].name
    out_path = args.output_dir / f"{file_name}_style.txt"
    file = out_path.open("w")
    file.write(style_string)
    file.close()
