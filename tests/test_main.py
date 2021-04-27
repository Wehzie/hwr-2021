import unittest

from src import main


class TestMain(unittest.TestCase):
    def test_hello_world(self):
        """
        Test that testing hello world from main works.
        """
        self.assertEqual(main.hello_world(), "hello_world")


# make tests runnable from the command line
if __name__ == "__main__":
    unittest.main()
