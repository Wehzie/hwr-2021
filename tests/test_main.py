import unittest
import os
from pathlib import Path

from dotenv import load_dotenv

from src.main import pipeline_train, pipeline_test

load_dotenv()


class TestMain(unittest.TestCase):
    def train_pipeline(self):
        """
        Test that the train pipeline works.
        """
        pipeline_train(train_fast=True)

    def test_pipeline(self):
        """
        Test that the prediction pipeline works.
        """
        input_dir = Path(os.environ["DATA_PATH"]) / "fragments" / "dev"
        output_dir = Path(os.environ["DATA_PATH"]) / "out"
        pipeline_test(input_dir, output_dir)


# make tests runnable from the command line
if __name__ == "__main__":
    unittest.main()
