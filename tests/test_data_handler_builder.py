import unittest

from src.data_handler.dataset_builder import DatasetBuilder


class TestDataHandlerBuilder(unittest.TestCase):
    def test_data_build_integration(self):
        builder = DatasetBuilder()

        if not builder.assert_data_correct():
            builder.download_all_data()
            builder.unpack_rename_data()
            builder.split_data_characters()
            builder.split_data_fragments()
            builder.create_font_data()

        self.assertTrue(builder.assert_data_correct(), "Data could not be built.")


# make tests runnable from the command line
if __name__ == "__main__":
    unittest.main()
