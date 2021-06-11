import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tap import Tap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras

file = Path(__file__).resolve()
project_root = file.parents[2]
sys.path.append(str(project_root))

from src.data_handler.hebrew import HebrewStyles
from model import StyleClassifierModel


class ArgParser(Tap):
    """
    Argument parser for the style classifier trainer.
    """

    data_path: Path = Path("data/style-data")  # input data folder
    plot_path: Path  # file to save training history plot to
    df_path: Path  # dataframe with the output of the model

    def configure(self) -> None:
        self.add_argument("--plot_path", required=False)
        self.add_argument("--df_path", required=False)


def plot_history(history, plot_path: Path) -> None:
    """
    Plot and save the history of the classifier training.
    """

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig(plot_path)


def main() -> None:
    ap = ArgParser()
    args = ap.parse_args()

    x = []
    y = []
    img_size = (60, 40)  # TODO: think about correct resizing method
    cv_img_size = (img_size[1], img_size[0])

    print(f"Loading jpeg images from {args.data_path}")
    for file_path in args.data_path.glob("**/*.jpg"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, cv_img_size)
        x.append((img, file_path.parent.name))
        y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train_img = np.array([x[0] for x in x_train])
    x_test_img = np.array([x[0] for x in x_test])
    x_test_char = np.array([x[1] for x in x_test])
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Training and validating on characters.")
    style_model = StyleClassifierModel()
    style_model.set_model(img_size, 0.2)
    style_model.model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    es = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        min_delta=0.007,
    )

    class_weights = dict(
        enumerate(
            class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
        )
    )

    # Train the model
    history = style_model.model.fit(
        x_train_img,
        y_train,
        validation_data=(x_test_img, y_test),
        epochs=50,
        class_weight=class_weights,
        callbacks=[es],
    )

    if args.plot_path is not None:
        print(f"Saving history plot to {args.plot_path}")
        plot_history(history, args.plot_path)

    # Confusion matrix on test data with final model
    y_pred = np.argmax(style_model.predict(x_test_img), axis=1)

    print(
        pd.crosstab(
            pd.Series(HebrewStyles.style_li[y] for y in y_test),
            pd.Series(HebrewStyles.style_li[y] for y in y_pred),
            rownames=["True:"],
            colnames=["Predicted:"],
            margins=True,
        )
    )

    if args.df_path is not None:
        print(f"Saving dataframe to {args.df_path}")
        df = pd.DataFrame(data={"char": x_test_char, "true": y_test, "pred": y_pred})
        df.to_pickle(args.df_path)


if __name__ == "__main__":
    main()
