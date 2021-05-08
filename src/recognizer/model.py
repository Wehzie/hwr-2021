import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, Input

class RecognizerModel:
    def __init__(self):
        self.model = None

    @staticmethod
    def __create_model(image_size, drop_rate) -> models.Model:
        model = models.Sequential()

        #conv1
        model.add(layers.Conv2D(16, 
                                kernel_size = (3, 3), 
                                activation='relu', 
                                input_shape=(image_size, image_size, 3)))
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
        model.add(layers.Dense(units = 500,
                               activation='relu',
                                ))

        #FCL2
        model.add(layers.Dense(units = 250,
                               activation='relu',
                                ))

        #FCL3
        model.add(layers.Dense(units = 100,
                               activation='relu',
                                ))

        #Softmax-layer (output)
        model.add(layers.Dense(units = 27,
                               activation='softmax',
                                ))
        return model

    def get_summary(self) -> str:
        return self.model.summary()

    def predict(self, data) -> np.ndarray:
        return self.model.predict(data)

    def save_model(self, model_path: Path) -> None:
        self.model.save(model_path)

    def load_model(self, model_path: Path) -> None:
        self.model = models.load_model(model_path, compile=False)