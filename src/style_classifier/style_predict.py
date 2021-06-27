import os
import sys
import numpy as np
import cv2 as cv
from pathlib import Path
from tap import Tap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_handler.hebrew import HebrewStyles
from src.style_classifier.model import StyleClassifierModel
from src.style_classifier.style_train import CV_IMG_SIZE


class ArgParser(Tap):
    """Argument parser for the style predictor."""

    input_dir: Path  # Directory that contains segmented characters
    output_dir: Path  # Directory where prediction will be saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("input_dir")
        self.add_argument("output_dir")


def date_frag(input_dir: Path, output_dir: Path, model_name: str = "model_0") -> None:
    """
    Date a single fragment.

    input_dir: Path to a single directory with extracted characters.
        For example data/fragment/characters/
        Where character contains instances of character_L*_W*_C*.png
    output_dir: Path to a directory to save the date prediction.
    returns: None. Saves to file.
    """
    imgs = []
    for file_path in input_dir.glob("*.png"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, CV_IMG_SIZE)
        imgs.append(img)
    imgs = np.array(imgs)

    style_model = StyleClassifierModel()
    style_model.load_model(model_name)

    preds_sum = np.sum(style_model.predict(imgs), axis=0)
    print("Style certainties:", preds_sum / len(imgs))
    style_pred = np.argmax(preds_sum)
    style_string = HebrewStyles.style_li[style_pred]

    file_name = input_dir.parents[0].name
    out_path = output_dir / f"{file_name}_style.txt"
    file = out_path.open("w")
    file.write(style_string)
    file.close()


def match_character_folders(input_dir: Path) -> list:
    """
    Find folders named "characters" under the provided path.

    input_dir: Search under this path.
    return: List of Paths matching the query.
    """
    # segmentation saves characters under fragment/characters
    return input_dir.glob("**/characters")


def date_fragments(input_dir: Path, output_dir: Path) -> None:
    """
    Classify the epoch of one or multiple fragments.

    input_dir: Path to a directory where extracted characters from
        multiple fragments are matched and transcribed.
        For example data/multiple_fragments
        May contain frag1/chars and frag2/chars
        Both of which will be dated
    output_dir: Path to a directory to save the epoch prediction.
    returns: None. Saves to file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    char_dirs: list = match_character_folders(input_dir)
    for char_dir in char_dirs:
        date_frag(char_dir, output_dir)


if __name__ == "__main__":
    args = ArgParser(description="Predict style from segmented characters").parse_args()
    assert args.input_dir.exists(), "Input directory does not exist"
    assert args.output_dir.exists(), "Output directory does not exist"

    date_fragments(args.input_dir, args.output_dir)
