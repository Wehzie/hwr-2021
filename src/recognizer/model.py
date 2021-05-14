import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, Input

class RecognizerModel:
    def __init__(self):
        """
        Initializes the recognizer model
        """
        self.model = None

    def create_model(self, image_size = (50, 60), drop_rate = 0.3) -> models.Model:
        """
        Creates the sequential CNN for character recognition
        """
        model = models.Sequential()

        #conv1
        model.add(layers.Conv2D(16, 
                                kernel_size = (3, 3), 
                                activation='relu', 
                                input_shape=(image_size[0], image_size[1], 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(drop_rate))

        #conv2
        model.add(layers.Conv2D(32, 
                                kernel_size = (3, 3), 
                                activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(drop_rate))

        #conv3
        model.add(layers.Conv2D(64, 
                                kernel_size = (3, 3), 
                                activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(drop_rate))

        #FCL1
        model.add(layers.Flatten())
        model.add(layers.Dense(units = 640,
                               activation='relu',
                                ))

        #FCL2
        model.add(layers.Dense(units = 320,
                               activation='relu',
                                ))

        #FCL3
        model.add(layers.Dense(units = 160,
                               activation='relu',
                                ))

        #Softmax-layer (output)
        model.add(layers.Dense(units = 27,
                               activation='softmax',
                                ))
        return model

    def get_summary(self) -> str:
        """
        Get summary of the recognizer model
        """
        return self.model.summary()

    def predict(self, data) -> np.ndarray:
        """
        Predicts the character labels of given input images
        """
        return self.model.predict(data)

    def save_model(self, model_path: Path) -> None:
        """
        Saves the model to model_path
        """
        self.model.save(model_path)

    def load_model(self, model_path: Path) -> None:
        """
        Loads a saved model from model_path
        """
        self.model = models.load_model(model_path, compile=False)