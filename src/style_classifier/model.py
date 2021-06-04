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
            # conv1
            model = layers.Conv2D(
                16,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(image_size[0], image_size[1], 3),
            )
            model = layers.MaxPooling2D((2, 2))(model)
            model = layers.Dropout(drop_rate)(model)

            # conv2
            model = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(model)
            model = layers.MaxPooling2D((2, 2))(model)
            model = layers.Dropout(drop_rate)(model)

            # conv3
            model = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(model)
            model = layers.MaxPooling2D((2, 2))(model)
            model = layers.Dropout(drop_rate)(model)

            # FCL1
            model = layers.Flatten()(model)
            return model

        images_input = layers.Input(shape=image_size + (3,))

        # selecting CNN architecture
        if arch == "custom":
            images_model = cnn_custom()(images_input)
        else:
            images_model = cnn_dense_net_121()(images_input)

        images_model = layers.Dense(units=512, activation="relu")(images_model)
        images_model = layers.Dropout(drop_rate)(images_model)

        char_input = layers.Input(shape=(27,))
        char_model = layers.Dense(units=128, activation="relu")(char_input)
        char_model = layers.Dropout(drop_rate)(char_model)

        merged = layers.concatenate([images_model, char_model])

        merged = layers.Dense(units=512, activation="relu")(merged)
        merged = layers.Dropout(drop_rate)(merged)

        # Softmax-layer (output)
        output = layers.Dense(units=3, activation="softmax")(merged)

        self.model = models.Model(
            inputs=[images_input, char_input],
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
