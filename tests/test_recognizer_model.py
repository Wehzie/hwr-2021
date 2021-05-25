import unittest

from src.recognizer.model import RecognizerModel


class TestRecognizerModel(unittest.TestCase):
    def test_create_model_default(self):
        Char_Recognizer = RecognizerModel()
        Char_Recognizer.set_model()
        self.assertIsNotNone(
            Char_Recognizer.model, "DenseNet121 CNN could not be built."
        )

    def test_create_model_custom(self):
        Char_Recognizer = RecognizerModel()
        Char_Recognizer.set_model(arch="custom")
        self.assertIsNotNone(Char_Recognizer.model, "Custom CNN could not be built.")


# make tests runnable from the command line
if __name__ == "__main__":
    unittest.main()
