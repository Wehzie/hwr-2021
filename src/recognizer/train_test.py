import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.data_handler.dataset_builder import DatasetBuilder

dataset_builder = DatasetBuilder()
print("test")
#https://www.tensorflow.org/tutorials/images/cnn

def load_data():
    NotImplemented

def train_model():
    NotImplemented
    # _train, y_train, X_dev, y_dev, X_test, y_test

def test_model():
    NotImplemented

