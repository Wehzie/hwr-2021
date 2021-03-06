import os
import sys
from pathlib import Path

# NOTE: Why are all TensorFlow messages suppressed?
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from tap import Tap
from tensorflow import keras

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.hebrew import HebrewStyles
from src.style_classifier.model import StyleClassifierModel

IMG_SIZE = (70, 60)
CV_IMG_SIZE = (IMG_SIZE[1], IMG_SIZE[0])


class ArgParser(Tap):
    """Argument parser for the style classifier trainer."""

    input_path: Path = Path("data/character_styles")  # input data folder
    test_split: bool = False
    save_all: bool = False  # save all below to model directory
    save_test_indices: Path  # if given, where numpy array with test indices is saved
    save_plot: Path  # if given, where training history plot image is saved
    save_df: Path  # if given, where dataframe with model output is saved

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("-i", "--input_path")
        self.add_argument("-s", "--test_split", action="store_true", required=False)
        self.add_argument("-a", "--save_all", action="store_true", required=False)
        self.add_argument("-t", "--save_test_indices", required=False)
        self.add_argument("-p", "--save_plot", required=False)
        self.add_argument("-d", "--save_df", required=False)


def plot_history(history, plot_path: Path) -> None:
    """Plot and save the history of the classifier training."""
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig(plot_path)


def load_data(input_path: Path) -> tuple:
    """
    Load the style data.
    Return data as X and labels as y.
    """
    x = []  # character images
    y = []  # character labels

    print(f"Loading jpeg images from {input_path}")
    for file_path in input_path.glob("**/*.jpg"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, CV_IMG_SIZE)
        x.append((img, file_path.parent.name))
        y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

    y = np.array(y)
    x = np.array([x[0] for x in x])

    return x, y


def load_data_split(
    input_path: Path,
    save_all: bool,
    model_name: str,
    save_test_indices: Path = None,
    test_size: float = 0.33,
) -> tuple:
    """
    Load the style data.
    Split into train and test.
    Return data as X and labels as y.
    """
    x = []  # character images
    y = []  # character labels

    print(f"Loading jpeg images from {input_path}")
    for file_path in input_path.glob("**/*.jpg"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, CV_IMG_SIZE)
        x.append((img, file_path.parent.name))
        y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

    y = np.array(y)
    x_img = np.array([x[0] for x in x])
    x_char = np.array([x[1] for x in x])

    # data split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, test_index in strat_split.split(x, y):
        x_train_img, x_test_img = x_img[train_index], x_img[test_index]
        x_test_char = x_char[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_idx = test_index
        if save_all:
            save_test_indices = (
                Path(os.environ["STYLE_MODEL_DATA"]) / model_name / "indices"
            )
            save_test_indices.mkdir(parents=True, exist_ok=True)
        if save_test_indices is not None:
            np.save(save_test_indices, np.array(test_idx))

    return x_train_img, y_train, x_test_img, x_test_char, y_test


def initialize_model(model_arch: str = "dense_net_121") -> StyleClassifierModel:
    """Initialize the style classifier."""
    style_model = StyleClassifierModel()
    style_model.set_model(IMG_SIZE, 0.2, model_arch)
    style_model.model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return style_model


def model_analysis(
    history,
    style_model: StyleClassifierModel,
    x_test_img,
    x_test_char,
    y_test,
    save_all: bool,
    save_plot: bool,
    save_df: bool,
    model_name: str,
) -> None:
    """Analyse the model and save analyses to file."""
    if save_all:
        save_plot = Path(os.environ["STYLE_MODEL_DATA"]) / model_name / "history.png"
        save_df = Path(os.environ["STYLE_MODEL_DATA"]) / model_name / "dataframe"

    if save_plot is not None:
        print(f"Saving history plot to {save_plot}")
        plot_history(history, save_plot)

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

    if save_df is not None:
        print(f"Saving dataframe to {save_df}")
        df = pd.DataFrame(data={"char": x_test_char, "true": y_test, "pred": y_pred})
        df.to_pickle(save_df)


def train_test_style_classifier(args, model_arch: str = "dense_net_121") -> None:
    """
    Train and test the style classifier.
    Splits a data set into train and test.
    Analyses the performance on the test set.
    """

    # initialize model
    style_model = initialize_model(model_arch)
    model_name = style_model.get_model_name()

    # load data
    (x_train_img, y_train, x_test_img, x_test_char, y_test) = load_data_split(
        args.input_path, args.save_all, model_name, args.save_test_indices
    )

    # set training parameters
    es = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        min_delta=0.007,
    )

    # balance data set
    class_weights = dict(
        enumerate(
            class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
        )
    )

    # Train the model
    print(f"Training and validating on {len(x_train_img)} characters.")
    history = style_model.model.fit(
        x_train_img,
        y_train,
        validation_data=(x_test_img, y_test),
        epochs=15,
        class_weight=class_weights,
        callbacks=[es],
    )

    # analyse and save run
    model_analysis(
        history,
        style_model,
        x_test_img,
        x_test_char,
        y_test,
        args.save_all,
        args.save_plot,
        args.save_df,
        model_name,
    )

    style_model.save_model(model_name)


def train_style_classifier(
    input_path: Path = None, model_arch: str = "dense_net_121"
) -> None:
    """Train the style classifier on all available data."""

    if input_path == None:
        input_path = Path(os.environ["DATA_PATH"]) / "character_styles"

    # load data
    (x, y) = load_data(input_path)

    # initialize model
    style_model = initialize_model(model_arch)

    # balance data set
    class_weights = dict(
        enumerate(
            class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
        )
    )

    # Train the model
    print(f"Training on {len(x)} characters.")
    style_model.model.fit(
        x,
        y,
        epochs=15,
        class_weight=class_weights,
    )

    model_name = style_model.get_model_name()
    style_model.save_model(model_name)


if __name__ == "__main__":
    ap = ArgParser()
    args = ap.parse_args()
    assert args.input_path.exists(), "Input directory does not exist"

    if args.test_split:
        train_test_style_classifier(args)
    else:
        train_style_classifier(args.input_path)
