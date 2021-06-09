import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras

file = Path(__file__).resolve()
parent, project_root = file.parent, file.parents[2]
sys.path.append(str(project_root))

from src.data_handler.hebrew import HebrewStyles
from model import StyleClassifierModel

data_path = Path(project_root) / "data" / "style-data"

img_size = (60, 40)  # TODO: think about correct resizing method
cv_img_size = (img_size[1], img_size[0])

def plot_history(history):
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig("plot.png")

def main():
    X = []
    y = []

    for file_path in data_path.glob("**/*.jpg"):
        img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
        img = cv.resize(img, cv_img_size)
        X.append((img, file_path.parent.name))
        y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

    y = np.array(y)
    X_img = np.array([x[0] for x in X])
    X_char = np.array([x[1] for x in X])

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
    for train_index, test_index in strat_split.split(X, y):
        X_train_img, X_test_img = X_img[train_index], X_img[test_index]
        X_train_char, X_test_char = X_char[train_index], X_char[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_idx = test_index

    np.save("data/model/style-classifier/test_indices", np.array(test_idx))

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
        patience=2,
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
        X_train_img,
        y_train,
        validation_data=(X_test_img, y_test),
        epochs=25,
        class_weight=class_weights,
        callbacks=[es],
    )

    #style_model.save_model("style-classifier")

    plot_history(history)

    # Confusion matrix on test data with final model
    y_pred = np.argmax(style_model.predict(X_test_img), axis=1)

    print(
        pd.crosstab(
            pd.Series(HebrewStyles.style_li[y] for y in y_test),
            pd.Series(HebrewStyles.style_li[y] for y in y_pred),
            rownames=["True:"],
            colnames=["Predicted:"],
            margins=True,
        )
    )

    df = pd.DataFrame(data={"char": X_test_char, "true": y_test, "pred": y_pred})
    df.to_pickle("style_output_df.pkl")

if __name__ == "__main__":
    main()
