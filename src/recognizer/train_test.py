import os, sys, inspect, cv2
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from glob import glob
from src.data_handler.dataset_builder import DatasetBuilder
from src.recognizer.model import RecognizerModel
from dotenv import load_dotenv
from pathlib import Path
from tensorflow import keras

#https://www.tensorflow.org/tutorials/images/cnn


class train_test():

    load_dotenv() 

    def __init__(self) -> None:
        self.dataset_builder = DatasetBuilder()
        self.dataset_builder.create_font_data()
        self.letters = ['Alef', 'Ayin', 'Bet', 'Dalet', 'Gimel', 'He', 'Het', 'Kaf',
                        'Kaf-final', 'Lamed', 'Mem', 'Mem-medial', 'Nun-final', 'Nun-medial',
                        'Pe', 'Pe-final', 'Qof', 'Resh', 'Samekh', 'Shin', 'Taw',
                        'Tet', 'Tsadi-final', 'Tsadi-medial', 'Waw', 'Yod', 'Zayin']
        self.img_size = (60,70)
        self.X_pretrain, self.y_pretrain, self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test = self.load_data()

    def load_data(self):
        read_path = Path(os.environ['DATA_PATH']) / self.dataset_builder.rename_folders['new'][1]
        pretrain_path = Path(os.environ['FONT_DATA'] + "training")
        if not self.dataset_builder.assert_data_correct():
                self.dataset_builder.download_all_data()
                self.dataset_builder.unpack_rename_data()
                self.dataset_builder.split_data()
        X_pretrain, y_pretrain, X_train, y_train, X_dev, y_dev, X_test, y_test = [], [], [],[],[],[],[],[]
        img_size = (self.img_size[1], self.img_size[0])
        
        # pretraining_data
        for i in range(len(glob(f'{pretrain_path}/*.jpeg'))):
            img = glob(f'{pretrain_path}/*.jpeg')[i]
            image = cv2.imread(img)
            image = cv2.resize(image, img_size)
            X_pretrain.append(image)
            y_pretrain.append(i)

        for letter in self.letters: 
            train_images = glob(f'{Path(read_path/"train"/letter)}/*.pgm')
            dev_images = glob(f'{Path(read_path/"dev"/letter)}/*.pgm')
            test_images = glob(f'{Path(read_path/"test"/letter)}/*.pgm')

            # training data
            for img in train_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
                X_train.append(image)
                y_train.append(self.letters.index(letter))

            # dev data
            for img in dev_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
                X_dev.append(image)
                y_dev.append(self.letters.index(letter))

            # test data
            for img in test_images:
                image = cv2.imread(img)
                image = cv2.resize(image, img_size)
                X_test.append(image)
                y_test.append(self.letters.index(letter))

        return np.array(X_pretrain), np.array(y_pretrain), np.array(X_train), np.array(y_train), np.array(X_dev), np.array(y_dev), np.array(X_test), np.array(y_test)


    def train_model(self):
        Char_Recognizer = RecognizerModel()
        Char_Recognizer.model = Char_Recognizer.create_model(self.img_size, 0.3)
        Char_Recognizer.model.compile(optimizer=keras.optimizers.Adam() , loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        #print(Char_Recognizer.get_summary())
        
        # print("pretraining on font data...")
        # Char_Recognizer.model.fit(self.X_pretrain, self.y_pretrain) #pretraining
        
        es = keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 2, restore_best_weights= True, min_delta= 0.007)
        print("training and validating on characters...")
        Char_Recognizer.model.fit(self.X_train, self.y_train, validation_data=(self.X_dev,self.y_dev), epochs=8, callbacks = [es])
        
        print(Char_Recognizer.get_summary())
        # Confusion matrix on dev data with final model
        y_pred = Char_Recognizer.predict(self.X_dev)
        y_predict = np.argmax(y_pred, axis = 1)
        print(pd.crosstab(pd.Series(self.y_dev), pd.Series(y_predict), rownames = ["True:"], colnames = ["Predicted:"], margins = True))

    def test_model(self):
        Char_Recognizer = RecognizerModel()
        Char_Recognizer.model = Char_Recognizer.create_model(self.img_size, 0.4)
        Char_Recognizer.model.compile(optimizer=keras.optimizers.Adam() , loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        print(Char_Recognizer.get_summary())
        Char_Recognizer.model.fit(self.X_train, self.y_train, validation_data=(self.X_test,self.y_test), epochs=5)

if __name__ == "__main__":
    trainer = train_test()
    trainer.train_model()
