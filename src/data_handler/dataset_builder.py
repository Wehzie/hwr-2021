import glob
import os
import re
import requests
import shutil
from pathlib import Path

from dotenv import load_dotenv

class DatasetBuilder():

    load_dotenv()   # load environment variables from .env file

    chunk_size = 128    #chunk size for downloading data

    rename_folders = {
        "original": [
            "image-data.zip",
            "monkbrill.tar.gz",
            "characters_for_style_classification.zip",
            "full_images_periods",
            ],
        "new": ["fragments", "characters", "character_styles", "fragment_styles"]
    }

    data_split = {
        "train": 0.8,
        "dev": 0.1,
        "test": 0.1,
    }

    directory_size = {
        "characters": 27,
    }
    
    def __init__(self):
        pass

    def download_data_nextcloud(self, token: str):
        '''
        Download data from a single URL using the requests module.
        Only works with Nextcloud instances via WebDAV.
        '''

        # download data
        r = requests.get(os.environ['NC_WEBDAV_URL'], auth=(token, os.environ['NC_PASSWORD']))

        # find file name
        d = r.headers['content-disposition']
        f_name = re.findall('filename="(.+)"', d)[0]

        # save file
        with open(Path(os.environ['DATA_PATH'])/f_name, 'wb') as f:
            for chunk in r.iter_content(self.chunk_size):
                f.write(chunk)

    def download_all_data(self):
        '''
        Download the character-images and fragment images data.
        '''
        self.download_data_nextcloud(os.environ['NC_TOKEN_TRAIN_CHARACTERS'])
        self.download_data_nextcloud(os.environ['NC_TOKEN_TRAIN_FRAGMENTS'])
        self.download_data_nextcloud(os.environ['NC_TOKEN_TRAIN_CHARACTER_STYLE'])
        self.download_data_nextcloud(os.environ['NC_TOKEN_TRAIN_FRAGMENT_STYLE'])

    def unpack_rename_data(self):
        '''
        Unpack and rename image-data.zip, monkbrill.tar.gz,
        characters_for_style_classification.zip and full_images_periods.zip.
        Then delete each original file.
        '''
        read_path = Path(os.environ['DATA_PATH']) / self.rename_folders['original'][0]
        write_path = Path(os.environ['DATA_PATH']) / self.rename_folders['new'][0]
        shutil.unpack_archive(read_path, write_path, 'zip')
        os.remove(read_path)    # delete original file

        read_path = Path(os.environ['DATA_PATH']) / self.rename_folders['original'][1]
        write_path = Path(os.environ['DATA_PATH']) / self.rename_folders['new'][1]
        shutil.unpack_archive(read_path, write_path, 'gztar')
        os.remove(read_path)

    def split_data(self) -> None:
        '''
        Split data into train, dev (validation) and test sets.
        '''
        read_path = Path(os.environ['DATA_PATH']) / self.rename_folders['new'][1]
        letter_directories = os.listdir(read_path)
        for data_set in self.data_split: # train, dev, test
            os.mkdir(Path(read_path/data_set))
            for letter in letter_directories:
                os.mkdir(Path(read_path/data_set/letter))
        for letter in letter_directories:
            images = os.listdir(Path(read_path/letter))
            split_train = int(len(images)*self.data_split["train"])
            split_dev = split_train + int(len(images)*self.data_split["dev"])

            train_images = images[:split_train]
            dev_images = images[split_train:split_dev]
            test_images = images[split_dev:]

            for image in train_images:
                shutil.move(Path(read_path/letter/image), Path(read_path/"train"/letter/image))
            for image in dev_images:
                shutil.move(Path(read_path/letter/image), Path(read_path/"dev"/letter/image))
            for image in test_images:
                shutil.move(Path(read_path/letter/image), Path(read_path/"test"/letter/image))
            os.rmdir(Path(read_path/letter))

    def assert_data_correct(self, data_type = "characters") -> bool:
        '''
        Assert that the data exists and is in the correct format.
        '''
        read_path = Path(os.environ['DATA_PATH']) / self.rename_folders['new'][1]
        if not os.path.exists(read_path):
            return False
        if set(os.listdir(read_path)) != {"train", "dev", "test"}:
            return False
        for data_set in os.listdir(read_path):
            if len(os.listdir(Path(read_path/data_set))) != self.directory_size[data_type]:
                return False
        return True

if __name__ == "__main__":
    dataset_builder = DatasetBuilder()