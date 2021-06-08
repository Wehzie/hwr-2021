import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras

file = Path(__file__).resolve()
parent, project_root = file.parent, file.parents[2]
sys.path.append(str(project_root))

from src.data_handler.hebrew import HebrewStyles
from model import StyleClassifierModel

data_path = Path(project_root) / "data" / "style-data"


def plot_history(history):
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig("plot.png")


X = []
y = []
img_size = (60, 40)  # TODO: think about correct resizing method
cv_img_size = (img_size[1], img_size[0])

for file_path in data_path.glob("**/*.jpg"):
    img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
    img = cv.resize(img, cv_img_size)
    X.append((img, file_path.parent.name))
    y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

# print(np.mean([img.shape[0] for img in X]), np.mean([img.shape[1] for img in X]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_img = np.array([x[0] for x in X_train])
X_train_char = np.array([x[1] for x in X_train])
X_test_img = np.array([x[0] for x in X_test])
X_test_char = np.array([x[1] for x in X_test])
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

# es = keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     patience=2,
#     restore_best_weights=True,
#     min_delta=0.007,
# )

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
    class_weight=class_weights
    # callbacks=[es],
)

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
