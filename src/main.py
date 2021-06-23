import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1].resolve()))

from src.recognizer.train_test import TrainTest

parser = argparse.ArgumentParser(description='Control the pipeline.')
parser.add_argument('--train', action='store_true',
                    help='train pipeline for character recognizer and style classifier')
parser.add_argument('--test', action='store_true',
                    help='test pipeline for character recognizer and style classifier')
args = parser.parse_args()

def pipeline_train():
    trainer = TrainTest()
    trainer.train_model()
    # TODO: style

def pipeline_test():
    pass

if __name__ == "__main__":
    if args.train:
        pipeline_train()
    if args.test:
        pipeline_test()
