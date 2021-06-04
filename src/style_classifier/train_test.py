import inspect
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow import keras

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.data_handler.dataset_builder import DatasetBuilder
from src.data_handler.hebrew import HebrewAlphabet
from src.recognizer.model import RecognizerModel

# https://www.tensorflow.org/tutorials/images/cnn


class TrainTest:
    """
    Train the hebrew character recognizer.
    """

    load_dotenv()

    def __init__(self) -> None:
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
        read_path = Path(os.environ["DATA_PATH"]) / "characters"
        pretrain_path = Path(os.environ["FONT_DATA"]) / "training"
        if not self.dataset_builder.assert_data_correct():
            self.dataset_builder.download_all_data()
            self.dataset_builder.unpack_rename_data()
            self.dataset_builder.split_data_characters()
            self.dataset_builder.create_font_data()
        X_pretrain, y_pretrain, X_train, y_train, X_dev, y_dev, X_test, y_test = tuple(
            [] for l in range(8)
        )
        img_size = (self.img_size[1], self.img_size[0])

        for letter in self.hebrew.letter_li:
            pretrain_images = glob(f"{Path(pretrain_path/letter)}/*.jpeg")
            train_images = glob(f'{Path(read_path/"train"/letter)}/*.pgm')
            dev_images = glob(f'{Path(read_path/"dev"/letter)}/*.pgm')
            test_images = glob(f'{Path(read_path/"test"/letter)}/*.pgm')

            # pretrain data
            for img in pretrain_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
                X_pretrain.append(image)
                y_pretrain.append(self.hebrew.letter_li.index(letter))

            # training data
            for img in train_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
                X_train.append(image)
                y_train.append(self.hebrew.letter_li.index(letter))

            # dev data
            for img in dev_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
                X_dev.append(image)
                y_dev.append(self.hebrew.letter_li.index(letter))

            # test data
            for img in test_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
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

    def train_model(self) -> None:
        """
        Train on training set (X_train, y_train) and
        validate on dev set (X_dev, y_dev).
        """
        self.recognizer.set_model(self.img_size, 0.3)
        self.recognizer.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        # print(self.recognizer.get_summary())

        print("Pretraining on font data.")
        self.recognizer.model.fit(self.X_pretrain, self.y_pretrain)  # pretraining

        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            min_delta=0.007,
        )
        print("Training and validating on characters.")
        self.recognizer.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_dev, self.y_dev),
            epochs=8,
            callbacks=[es],
        )

        print(self.recognizer.get_summary())
        # Confusion matrix on dev data with final model
        y_pred = self.recognizer.predict(self.X_dev)
        y_predict = np.argmax(y_pred, axis=1)
        print(
            pd.crosstab(
                pd.Series(self.y_dev),
                pd.Series(y_predict),
                rownames=["True:"],
                colnames=["Predicted:"],
                margins=True,
            )
        )

    def test_model(self) -> None:
        """
        Test a trained model on the traing set (X_train, y_train).
        """
        print(self.recognizer.get_summary())
        self.recognizer.model.evaluate(self.X_test, self.y_test)


if __name__ == "__main__":
    trainer = TrainTest()
    trainer.train_model()
