import inspect
import os
import re
import shutil
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root_dir)

from src.data_handler.font_images import FontImages


class DatasetBuilder:
    """
    Download and organize all required data for the pipeline.
    """

    load_dotenv()  # load environment variables from .env file

    chunk_size = 128  # chunk size for downloading data
    directory_size_characters = 27  # number of character variations in hebrew

    # old file name : new file name
    f_name_map = {
        "image-data.zip": "fragments",
        "monkbrill-jpg.tar.gz": "characters",
        "characters_for_style_classification.zip": "character_styles",
        "full_images_periods.zip": "fragment_styles",
        "habbakuk.ZIP": "font_characters",
    }

    # data subset name : fraction of the entire data
    data_split = {
        "train": 0.8,
        "dev": 0.1,
        "test": 0.1,
    }

    def __init__(self) -> None:
        self.hebrew_alphabet = None  # list of hebrew characters

    def download_data(self, url: str, source_type: str) -> None:
        """
        Download data from a single URL using the requests module.
        Only works with Nextcloud instances via WebDAV.
        """

        r = None  # request

        # download data from nextcloud
        if source_type == "nextcloud":
            token = url
            r = requests.get(
                os.environ["NC_WEBDAV_URL"], auth=(token, os.environ["NC_PASSWORD"])
            )

        # download data from generic URLs
        if source_type == "generic_url":
            s = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"
            }
            s.headers.update(headers)
            r = s.get(url)

        f_name = None  # file name

        if "content-disposition" in r.headers.keys():
            d = r.headers["content-disposition"]
            f_name = re.findall('filename="(.+)"', d)[0]
        else:
            f_name = url.split("/")[-1]

        # save file
        try:
            with open(Path(os.environ["DATA_PATH"]) / f_name, "wb") as f:
                for chunk in r.iter_content(self.chunk_size):
                    f.write(chunk)
        except OSError:
            print(f"Error: {os.listdir(Path(os.environ['DATA_PATH']))}")

    def download_all_data(self) -> None:
        """
        Download the character-images, fragment images and font data.
        """
        print("Download in progress.")
        self.download_data(os.environ["NC_TOKEN_TRAIN_CHARACTERS"], "nextcloud")
        self.download_data(os.environ["NC_TOKEN_TRAIN_FRAGMENTS"], "nextcloud")
        # self.download_data(os.environ["NC_TOKEN_TRAIN_CHARACTER_STYLE"], "nextcloud")
        # self.download_data(os.environ["NC_TOKEN_TRAIN_FRAGMENT_STYLE"], "nextcloud")
        self.download_data(os.environ["HABBAKUK_URL"], "generic_url")
        print("Download complete!")

    def unpack_rename_data(self):
        """
        Unpack and rename image-data.zip, monkbrill.tar.gz,
        characters_for_style_classification.zip, full_images_periods.zip
        and habbakuk.ZIP.
        Then delete each original file.
        """
        for old, new in self.f_name_map.items():
            try:
                read_path = Path(os.environ["DATA_PATH"]) / old
                write_path = Path(os.environ["DATA_PATH"]) / new
                if "ZIP" in old:
                    shutil.unpack_archive(read_path, write_path, "zip")
                else:
                    shutil.unpack_archive(read_path, write_path)
                os.remove(read_path)  # delete original file
            except OSError:
                print(f"File doest not exist at {read_path}")

    def split_data_characters(self) -> None:
        """
        Split labelled character data into train, dev (validation) and test sets.
        """
        read_path: Path = Path(os.environ["DATA_PATH"]) / "characters" 
        character_path: Path = Path(os.environ["DATA_PATH"]) / "characters" / "monkbrill2"
        for directory in os.listdir(character_path):
            shutil.move(Path(character_path / directory), read_path)
        os.rmdir(character_path)

        letter_directories: list = os.listdir(read_path)

        for subset in self.data_split:  # train, dev, test
            os.mkdir(Path(read_path / subset))
            for letter in letter_directories:
                os.mkdir(Path(read_path / subset / letter))

        for letter in letter_directories:
            images: list = os.listdir(Path(read_path / letter))
            split_train: int = int(len(images) * self.data_split["train"])
            split_dev: int = split_train + int(len(images) * self.data_split["dev"])

            train_images: tuple = (images[:split_train], "train")
            dev_images: tuple = (images[split_train:split_dev], "dev")
            test_images: tuple = (images[split_dev:], "test")
            split_indices: list = [train_images, dev_images, test_images]

            for subset in split_indices:
                for image in subset[0]:
                    shutil.move(
                        Path(read_path / letter / image),
                        Path(read_path / subset[1] / letter / image),
                    )

            os.rmdir(Path(read_path / letter))

    def assert_data_characters_correct(self) -> bool:
        """
        Assert that the labelled character data exists and is in the correct format.
        """
        read_path = Path(os.environ["DATA_PATH"]) / "characters"
        if not os.path.exists(read_path):
            return False
        if set(os.listdir(read_path)) != {"train", "dev", "test"}:
            return False
        for subset in os.listdir(read_path):
            if (
                len(os.listdir(Path(read_path / subset)))
                != self.directory_size_characters
            ):
                return False
        return True

    def assert_data_correct(self) -> bool:
        """
        Assert that all data exists and is in the correct format.
        """
        corr_char = self.assert_data_characters_correct()
        print("Character data correct?", corr_char)
        corr_font = FontImages().assert_data_correct()
        print("Font data correct?", corr_font)
        corr_frag = self.assert_data_fragments_correct()
        print("Frag data correc?", corr_frag)
        return True if corr_char and corr_font and corr_frag else False
    
    def split_data_fragments(self) -> None:
        """
        Split DSS fragments into train, dev (validation) and test sets.
        """
        read_path: Path = Path(os.environ["DATA_PATH"]) / "fragments"
        try:
            shutil.rmtree(read_path / "__MACOSX")
        except FileNotFoundError:
            print("Folder \"__MACOSX\" already removed.")
        
        for subset in self.data_split:  # make train, dev, test
            os.mkdir(Path(read_path / subset))
        
        # delete non-binarized images
        frags: list = os.listdir(read_path / "image-data")
        frags_binarized: list = [frag for frag in frags if "binarized" in frag]
        frags_delete: set = set(frags).difference(set(frags_binarized))
        for frag in frags_delete:
            os.remove(read_path / "image-data" / frag)
        frags = frags_binarized

        split_train: int = int(len(frags) * self.data_split["train"])
        split_dev: int = split_train + int(len(frags) * self.data_split["dev"])

        train_frags: tuple = (frags[:split_train], "train")
        dev_frags: tuple = (frags[split_train:split_dev], "dev")
        test_frags: tuple = (frags[split_dev:], "test")

        split_indices: list = [train_frags, dev_frags, test_frags]
        for subset in split_indices:
            for fragment in subset[0]:
                shutil.move(
                    Path(read_path / "image-data" / fragment),
                    Path(read_path / subset[1] / fragment),
                )

        os.rmdir(Path(read_path / "image-data"))    # delete empty folder
        
    def assert_data_fragments_correct(self) -> bool:
        """
        Assert that the fragment data exists and is in the correct format.
        """
        read_path = Path(os.environ["DATA_PATH"]) / "fragments"
        if not os.path.exists(read_path):
            return False
        if set(os.listdir(read_path)) != {"train", "dev", "test"}:
            return False
        for subset in os.listdir(read_path):
            # a file with "binarized" in the name should be in each subset
            if "binarized" not in os.listdir(Path(read_path / subset))[0]:
                return False
        return True

    def create_font_data(self):
        """
        Expects data/font_characters/Habbakuk.TTF
        """
        font_data = FontImages()
        if not font_data.assert_data_correct():
            font_data.create_images()


if __name__ == "__main__":
    data_build = DatasetBuilder()
    if not data_build.assert_data_correct():
        data_build.download_all_data()
        data_build.unpack_rename_data()
        data_build.split_data_characters()
        data_build.split_data_fragments()
        data_build.create_font_data()
