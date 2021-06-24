import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1].resolve()))

from src.data_handler.dataset_builder import DatasetBuilder
from src.recognizer.train_test import TrainTest
from src.style_classifier.style_train import ArgParser, train_style_classifier

parser = argparse.ArgumentParser(description='Control the pipeline.')
parser.add_argument('--train', action='store_true',
                    help='train pipeline for character recognizer and style classifier')
parser.add_argument('--test', action='store_true',
                    help='test pipeline for character recognizer and style classifier')
args = parser.parse_args()

def pipeline_train():
    """
    Download and preprocess all data.
    Train the character recognizer on all available data.
    Train the style classifier on all available data.
    """
    data_builder = DatasetBuilder()
    data_builder.build_data_set()
    char_rcg = TrainTest()
    char_rcg.train_model()
    train_style_classifier()

def pipeline_test():
    pass

if __name__ == "__main__":
    if args.train:
        pipeline_train()
    if args.test:
        pipeline_test()
