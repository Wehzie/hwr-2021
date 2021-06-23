import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tensorflow import keras
from tensorflow.keras import layers, models


class RecognizerModel:
    """
    Model for hebrew character recognition.
    """

    load_dotenv()

    def __init__(self):
        """
        Initializes the recognizer model.
        """
        self.model = None

    def set_model(
        self,
        image_size: tuple = (70, 60),
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
            old_model = keras.applications.DenseNet121(
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
            model.add(keras.layers.Flatten())

            model.add(keras.layers.Dropout(drop_rate))
            model.add(keras.layers.BatchNormalization())
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
            return model

        # selecting CNN architecture
        if arch == "custom":
            model = cnn_custom()
        else:
            model = cnn_dense_net_121()

        model.add(
            layers.Dense(
                units=640,
                activation="relu",
            )
        )
        model.add(keras.layers.Dropout(drop_rate))
        model.add(keras.layers.BatchNormalization())

        model.add(
            layers.Dense(
                units=160,
                activation="relu",
            )
        )
        model.add(keras.layers.Dropout(drop_rate))
        model.add(keras.layers.BatchNormalization())

        # Softmax-layer (output)
        model.add(
            layers.Dense(
                units=27,
                activation="softmax"
            )
        )
        self.model = model

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

    def get_model_name(self) -> str:
        """
        Create automatic incremental model names.
        For example, if model_0 exists, then model_1 will be returned.
        """
        path = Path(os.environ["MODEL_DATA"])
        # create data/model if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        # names of currently existing models
        names = [file.name for file in path.iterdir()]
        
        model_name = None
        for i in range(1000):
            if i == 999:
                print("Clear your model folder for more automatic name generation.")
            model_name = "model_" + str(i)
            if model_name in names: # this name exists
                continue
            else: # a new name is found
                break
        return model_name

    def save_model(self, model_name: str) -> None:
        """
        Saves the model to the model path under the given name.
        """
        # request a model name if none was provided
        if model_name == None:
            model_name = self.get_model_name()
        out_path = Path(os.environ["MODEL_DATA"]) / model_name
        # create data/model if it doesn't exist
        out_path.mkdir(parents=True, exist_ok=True) 
        self.model.save(out_path)

    def load_model(self, model_name: str) -> None:
        """
        Loads a saved model from the model path under the given name.
        """
        self.model = models.load_model(Path(os.environ["MODEL_DATA"]) / model_name)
