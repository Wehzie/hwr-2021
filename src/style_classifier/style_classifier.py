import inspect
from pathlib import Path
import os
import numpy as np
from tensorflow import keras
import cv2 as cv
import pandas as pd
from model import RecognizerModel
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))

data_path = Path(project_root_dir) / "data/style-data"

X = []
style_label = []
char_label = []
img_size = (40, 60)  # TODO: think about correct resizing method

for file_path in data_path.glob("**/*.jpg"):
    img = cv.imread(str(file_path.resolve()), cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, img_size)
    X.append(img)
    char_label.append(file_path.parents[0].name.lower())
    style_label.append(file_path.parents[1].name.lower())

# print(np.mean([img.shape[0] for img in X]), np.mean([img.shape[1] for img in X]))

X_train, X_test, y_train, y_test = train_test_split(
    X, style_label, test_size=0.33, random_state=42
)

X_train, X_test, y_train, y_test = (np.array(x) for x in train_test_split(
    X, style_label, test_size=0.33, random_state=42))

style_model = RecognizerModel()
style_model.set_model(img_size, 0.3)
style_model.model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
print(style_model.get_summary())

# es = keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     patience=2,
#     restore_best_weights=True,
#     min_delta=0.007,
# )
print("Training and validating on characters.")
style_model.model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=8,
    #callbacks=[es],
)

print(style_model.get_summary())
# Confusion matrix on test data with final model
y_pred = style_model.predict(X_test)
y_predict = np.argmax(y_pred, axis=1)
print(
    pd.crosstab(
        pd.Series(y_test),
        pd.Series(y_predict),
        rownames=["True:"],
        colnames=["Predicted:"],
        margins=True,
    )
)
