import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from tap import Tap

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_handler.hebrew import HebrewStyles

IMG_SIZE = (60, 40)  # TODO: think about correct resizing method
CV_IMG_SIZE = (IMG_SIZE[1], IMG_SIZE[0])


class ArgParser(Tap):
    """
    Argument parser for the style classifier trainer.
    """

    input_path: Path = Path("data/style-data")  # input data folder
    save_test_indices: Path  # if given, where numpy array with test indices is saved
    save_plot: Path  # if given, where training history plot image is saved
    save_df: Path  # if given, where dataframe with model output is saved

    def configure(self) -> None:
        self.add_argument("-i", "--input_path")
        self.add_argument("-t", "--save_test_indices", required=False)
        self.add_argument("-p", "--save_plot", required=False)
        self.add_argument("-d", "--save_df", required=False)


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

    print(f"Loading jpeg images from {args.input_path}")
    for file_path in args.input_path.glob("**/*.jpg"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, CV_IMG_SIZE)
        x.append((img, file_path.parent.name))
        y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

    y = np.array(y)
    x_img = np.array([x[0] for x in x])
    x_char = np.array([x[1] for x in x])

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
    for train_index, test_index in strat_split.split(x, y):
        x_train_img, x_test_img = x_img[train_index], x_img[test_index]
        x_test_char = x_char[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_idx = test_index
        if args.save_test_indices is not None:
            np.save(args.save_test_indices, np.array(test_idx))

    print("Training and validating on characters.")

    # Only import tf / keras after arguments are parsed
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow import keras

    from model import StyleClassifierModel

    style_model = StyleClassifierModel()
    style_model.set_model(IMG_SIZE, 0.2)
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
        epochs=25,
        class_weight=class_weights,
        callbacks=[es],
    )

    if args.save_plot is not None:
        print(f"Saving history plot to {args.save_plot}")
        plot_history(history, args.save_plot)

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

    if args.save_df is not None:
        print(f"Saving dataframe to {args.save_df}")
        df = pd.DataFrame(data={"char": x_test_char, "true": y_test, "pred": y_pred})
        df.to_pickle(args.save_df)


if __name__ == "__main__":
    main()
