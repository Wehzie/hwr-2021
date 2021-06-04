import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras

file = Path(__file__).resolve()
parent, project_root = file.parent, file.parents[2]
sys.path.append(str(project_root))

from src.data_handler.hebrew import HebrewAlphabet
from src.data_handler.hebrew import HebrewStyles
from model import StyleClassifierModel

data_path = Path(project_root) / "data" / "style-data"

X = []
y = []
img_size = (60, 40)  # TODO: think about correct resizing method
cv_img_size = (img_size[1], img_size[0])

for file_path in data_path.glob("**/*.jpg"):
    img = cv.imread(str(file_path.resolve()), cv.IMREAD_COLOR)
    img = cv.resize(img, cv_img_size)
    one_hot_char = np.zeros(27)
    one_hot_char[HebrewAlphabet.letter_li.index(file_path.parent.name)] = 1
    X.append([img, one_hot_char])
    y.append(HebrewStyles.style_li.index(file_path.parents[1].name))

# print(np.mean([img.shape[0] for img in X]), np.mean([img.shape[1] for img in X]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Extract the partial data from the split X_train and X_test
X_train_img = np.array([x[0] for x in X_train])
X_train_char = np.array([x[1] for x in X_train])
X_test_img = np.array([x[0] for x in X_test])
X_test_char = np.array([x[1] for x in X_test])

print("Training and validating on characters.")
style_model = StyleClassifierModel()
style_model.set_model(img_size, 0.5)
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
style_model.model.fit(
    [X_train_img, X_train_char],
    np.array(y_train),
    validation_data=([X_test_img, X_test_char], np.array(y_test)),
    epochs=10,
    class_weight=class_weights
    # callbacks=[es],
)

# Confusion matrix on test data with final model
y_pred = np.argmax(style_model.predict([X_test_img, X_test_char]), axis=1)
print(
    pd.crosstab(
        pd.Series(HebrewStyles.style_li[y] for y in y_test),
        pd.Series(HebrewStyles.style_li[y] for y in y_pred),
        rownames=["True:"],
        colnames=["Predicted:"],
        margins=True,
    )
)
