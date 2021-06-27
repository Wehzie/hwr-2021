import os
import sys
import numpy as np
import cv2 as cv
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model import StyleClassifierModel
from style_train import cv_img_size

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_handler.hebrew import HebrewStyles


if __name__ == "__main__":
    data_path = Path(PROJECT_ROOT) / "data" / "style-data"
    imgs = []
    true_style = []
    for file_path in data_path.glob("**/*.jpg"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, cv_img_size)
        imgs.append(img)
        true_style.append(HebrewStyles.style_li.index(file_path.parents[1].name))

    imgs = np.array(imgs)
    true_style = np.array(true_style)

    # if file_path.parents[1].name == "Herodian" and counter < 150:

    test_idx = np.load("data/model/style-classifier/test_indices.npy")
    imgs = imgs[test_idx]
    true_style = true_style[test_idx]

    style_model = StyleClassifierModel()
    style_model.load_model("style-classifier")

    for style in range(3):
        sub_imgs = imgs[np.where(true_style == style)]
        sub_true_style = true_style[np.where(true_style == style)]

        preds_sum = np.sum(style_model.predict(np.array(sub_imgs)), axis=0)
        style_pred = np.argmax(preds_sum)

        # print(HebrewStyles.style_li[style_pred])
        print(
            f"Style {style} predicted as {style_pred} with {preds_sum}"
            "confidence, {len(sub_imgs)} characters used."
        )
