import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.data_handler.dataset_builder import DatasetBuilder
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path

#https://www.tensorflow.org/tutorials/images/cnn


class train_test():

    load_dotenv() 

    def __init__(self) -> None:
        self.dataset_builder = DatasetBuilder()
        self.letters = ['Alef', 'Ayin', 'Bet', 'Dalet', 'Gimel', 'He', 'Het', 'Kaf',
                        'Kaf-final', 'Lamed', 'Mem', 'Mem-medial', 'Nun-final', 'Nun-medial',
                        'Pe', 'Pe-final', 'Qof', 'Resh', 'Samekh', 'Shin', 'Taw',
                        'Tet', 'Tsadi-final', 'Tsadi-medial', 'Waw', 'Yod', 'Zayin']
        self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test = self.load_data()

    def load_data(self):
        read_path = Path(os.environ['DATA_PATH']) / self.dataset_builder.rename_folders['new'][1]
        if not self.dataset_builder.assert_data_correct():
                self.dataset_builder.download_all_data()
                self.dataset_builder.split_data()
        data_sets = os.listdir(read_path)
        X_train, y_train, X_dev, y_dev, X_test, y_test = [],[],[],[],[],[]
        
        for letter in self.letters: 

            # training data
            for img in os.listdir(Path(read_path/data_sets[0]/letter)):
                X_train.append(Image.open(Path(read_path/data_sets[0]/letter/img)))
                y_train.append(self.letters.index(letter))

            # dev data
            for img in os.listdir(Path(read_path/data_sets[1]/letter)):
                X_dev.append(Image.open(Path(read_path/data_sets[1]/letter/img)))
                y_dev.append(self.letters.index(letter))

            # test data
            for img in os.listdir(Path(read_path/data_sets[2]/letter)):
                X_test.append(Image.open(Path(read_path/data_sets[2]/letter/img)))
                y_test.append(self.letters.index(letter))

        return X_train, y_train, X_dev, y_dev, X_test, y_test


    def train_model(self):
        NotImplemented
        # 

    def test_model(self):
        NotImplemented

if __name__ == "__main__":
    train_test()
