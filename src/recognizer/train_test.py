import os
import sys
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from dotenv import load_dotenv
from tensorflow import keras

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.dataset_builder import DatasetBuilder
from src.data_handler.hebrew import HebrewAlphabet
from src.recognizer.model import RecognizerModel


class TrainTest:
    """Train the Hebrew character recognizer."""

    load_dotenv()

    def __init__(self) -> None:
        """Initialize the TrainTest."""
        self.hebrew = HebrewAlphabet()
        self.dataset_builder = DatasetBuilder()
        self.recognizer = RecognizerModel()
        self.img_size = (60, 70)
        (
            self.X_pretrain,
            self.y_pretrain,
            self.X_train,
            self.y_train,
            self.X_dev,
            self.y_dev,
            self.X_test,
            self.y_test,
        ) = self.load_data()

    def load_data(self) -> tuple:
        """
        Populate X, y data pairs for pretraining, train, dev and test sets.
        """
        self.read_path = Path(os.environ["DATA_PATH"]) / "characters"
        self.pretrain_path = Path(os.environ["FONT_DATA"]) / "training"
        self.dataset_builder.build_data_set()
        X_pretrain, y_pretrain, X_train, y_train, X_dev, y_dev, X_test, y_test = tuple(
            [] for l in range(8)
        )

        for letter in self.hebrew.letter_li:
            pretrain_images = glob(f"{Path(self.pretrain_path/letter)}/*.jpeg")
            train_images = glob(f'{Path(self.read_path/"train"/letter)}/*.jpg')
            dev_images = glob(f'{Path(self.read_path/"dev"/letter)}/*.jpg')
            test_images = glob(f'{Path(self.read_path/"test"/letter)}/*.jpg')

            # pretrain data
            for img in pretrain_images:
                image = cv2.imread(img)
                image = cv2.resize(image, self.img_size)
                X_pretrain.append(image)
                y_pretrain.append(self.hebrew.letter_li.index(letter))

            # training data
            for img in train_images:
                image = cv2.imread(img)
                image = cv2.resize(image, self.img_size)
                X_train.append(image)
                y_train.append(self.hebrew.letter_li.index(letter))

            # dev data
            for img in dev_images:
                image = cv2.imread(img)
                image = cv2.resize(image, self.img_size)
                X_dev.append(image)
                y_dev.append(self.hebrew.letter_li.index(letter))

            # test data
            for img in test_images:
                image = cv2.imread(img)
                image = cv2.resize(image, self.img_size)
                X_test.append(image)
                y_test.append(self.hebrew.letter_li.index(letter))

        return (
            np.array(X_pretrain),
            np.array(y_pretrain),
            np.array(X_train),
            np.array(y_train),
            np.array(X_dev),
            np.array(y_dev),
            np.array(X_test),
            np.array(y_test),
        )

    def train_model(self, model_arch="dense_net_121") -> None:
        """Train on X_train, y_train and validate on X_dev, y_dev."""
        self.recognizer.set_model((self.img_size[1], self.img_size[0]), 0.3, model_arch)
        self.recognizer.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        font_chars = list((self.read_path / "train" / "Alef").iterdir())
        if len(font_chars) == 27:
            # print(self.recognizer.get_summary())
            print("Pretraining on font data.")
            self.recognizer.model.fit(self.X_pretrain, self.y_pretrain)  # pretraining

        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            min_delta=0.008,
        )
        print("Training and validating on characters.")
        history = self.recognizer.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_dev, self.y_dev),
            epochs=15,
            callbacks=[es],
        )

        print(self.recognizer.get_summary())
        model_name = self.recognizer.get_model_name()
        self.recognizer.save_model(model_name)

        self.model_analysis(model_name, history)

    def model_analysis(self, model_name: str, history) -> None:
        """Analyse how well a model performed."""
        # probabilites
        y_pred_prob = self.recognizer.predict(self.X_test)
        # most likely class
        y_pred = np.argmax(y_pred_prob, axis=1)
        # compare true and predicted classes on test set

        # path handling for writing to file
        output_dir = Path(os.environ["MODEL_DATA"]) / model_name
        out_name = "classification_report.txt"
        out_path = output_dir / out_name

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # plot accuracies and losses with respect to epochs
        plt.plot(epochs, acc, 'r', label='Train accuracy')
        plt.plot(epochs, val_acc, 'b', label='Val accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.savefig(output_dir/"acc-plot")

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Val loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(output_dir/"loss-plot")

        # create, print and write to file a sklearn classification report
        print(set(self.y_test) - set(y_pred))
        report = classification_report(self.y_test, y_pred)
        print(report)
        with open(out_path, "w") as f:
            f.write(report)

        self.make_heatmap(y_pred, output_dir)

    def make_heatmap(self, y_pred: np.ndarray, output_dir: Path) -> None:
        """
        Create a heatmap on correct character prediction frequency.

        y_pred: predicted class
        return: None. Saves figure
        """
        df = pd.crosstab(
            pd.Series(self.y_test),
            pd.Series(y_pred),
            rownames=["True:"],
            colnames=["Predicted:"],
            margins=True,
        )
        plt.figure()
        sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt="g", cmap="viridis")
        # path handling for writing to file
        out_name = "heatmap.png"
        out_path = output_dir / out_name
        plt.savefig(out_path)

    def train_full_model(self, model_arch="dense_net_121") -> None:
        """Train on all data and save the model, validate using test data"""
        X_concat = np.concatenate((self.X_train, self.X_dev, self.X_test), axis=0)
        y_concat = np.concatenate((self.y_train, self.y_dev, self.y_test), axis=0)
        self.recognizer.set_model((self.img_size[1], self.img_size[0]), 0.3, model_arch)
        self.recognizer.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        font_chars = list((self.read_path / "train" / "Alef").iterdir())
        if len(font_chars) == 27:
            # print(self.recognizer.get_summary())
            print("Pretraining on font data.")
            self.recognizer.model.fit(self.X_pretrain, self.y_pretrain)  # pretraining

        print("Training on characters.")
        self.recognizer.model.fit(
            X_concat,
            y_concat,
            epochs=6,
        )

        print(self.recognizer.get_summary())
        model_name = self.recognizer.get_model_name()
        self.recognizer.save_model(model_name)


if __name__ == "__main__":
    trainer = TrainTest()
    trainer.train_model()
