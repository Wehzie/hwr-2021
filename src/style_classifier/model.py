import os
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121


class StyleClassifierModel:
    """
    Model for Hebrew writing style recognition using individual characters.
    """

    load_dotenv()

    def __init__(self):
        """
        Initializes the recognizer model.
        """
        self.model = None

    def set_model(
        self,
        image_size: tuple = (40, 60),
        drop_rate: float = 0.3,
        arch: str = "dense_net_121",
    ) -> None:
        """
        Creates the sequential CNN for character recognition
        """

        def cnn_dense_net_121() -> models.Model:
            """
            DenseNet121 pretrained architecture.
            """
            old_model = DenseNet121(
                include_top=False,
                weights="imagenet",
                input_shape=(image_size[0], image_size[1], 3),
            )
            for layer in old_model.layers[:149]:
                layer.trainable = False
            for layer in old_model.layers[149:]:
                layer.trainable = True

            model = models.Sequential()

            model.add(old_model)
            model.add(layers.Flatten())
            model.add(layers.Dropout(drop_rate))
            model.add(layers.BatchNormalization())

            return model

        def cnn_custom() -> models.Model:
            """
            Custom CNN Architecture.
            """

            model = models.Sequential()

            # conv1
            model.add(
                layers.Conv2D(
                    16,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=(image_size[0], image_size[1], 3),
                )
            )
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(drop_rate))

            # conv2
            model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(drop_rate))

            # conv3
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(drop_rate))

            # FCL1
            model.add(layers.Flatten())
            model.add(layers.Dropout(drop_rate))
            model.add(layers.BatchNormalization())

            return model

        image_input = layers.Input(shape=image_size + (3,))

        # selecting CNN architecture
        if arch == "custom":
            model = cnn_custom()(image_input)
        else:
            model = cnn_dense_net_121()(image_input)

        model = layers.Dense(units=512, activation="relu")(model)
        model = layers.Dropout(drop_rate)(model)
        model = layers.BatchNormalization()(model)

        model = layers.Dense(units=256, activation="relu")(model)
        model = layers.Dropout(drop_rate)(model)
        model = layers.BatchNormalization()(model)

        # Softmax-layer (output)
        output = layers.Dense(units=3, activation="softmax")(model)

        self.model = models.Model(
            inputs=image_input,
            outputs=output,
        )

    def get_summary(self) -> str:
        """
        Get summary of the recognizer model.
        """
        return self.model.summary()

    def predict(self, data) -> np.ndarray:
        """
        Predicts the character labels of given input images.
        """
        return self.model.predict(data)

    def save_model(self, model_name: str) -> None:
        """
        Saves the model to the model path under the given name.
        """
        self.model.save(Path(os.environ["MODEL_DATA"]) / model_name)

    def load_model(self, model_name: str) -> None:
        """
        Loads a saved model from the model path under the given name.
        """
        self.model = models.load_model(Path(os.environ["MODEL_DATA"]) / model_name)
