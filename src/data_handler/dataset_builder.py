import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv
from scipy.sparse import data

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.font_images import FontImages
from src.data_handler.dss_augment import Augmenter


class DatasetBuilder:
    """Download and organize all required data for the pipeline."""

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
        """Initialize the dataset builder."""
        self.hebrew_alphabet = None  # list of hebrew characters
        self.augmenter = Augmenter()

    def download_data(self, url: str, source_type: str) -> None:
        """Download data from a single URL using the requests module.

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
            print(f"Error: {list(Path(os.environ['DATA_PATH']).iterdir())}")

    # TODO: Rename this method to download_char_recog_data()
    def download_all_data(self) -> None:
        """Download the character-images, fragment images and font data."""
        print("Download in progress.")
        self.download_data(os.environ["NC_TOKEN_TRAIN_CHARACTERS"], "nextcloud")
        self.download_data(os.environ["NC_TOKEN_TRAIN_FRAGMENTS"], "nextcloud")
        self.download_data(os.environ["HABBAKUK_URL"], "generic_url")
        print("Download complete!")
    
    def download_style_data(self) -> None:
        """Download the style character and fragment data."""
        print("Download in progress.")
        self.download_data(os.environ["NC_TOKEN_TRAIN_CHARACTER_STYLE"], "nextcloud")
        self.download_data(os.environ["NC_TOKEN_TRAIN_FRAGMENT_STYLE"], "nextcloud")

    def unpack_rename_data(self):
        """Unpack and rename downloaded data.

        Unpack and rename image-data.zip, monkbrill.tar.gz,
        characters_for_style_classification.zip, full_images_periods.zip
        and habbakuk.ZIP. Then delete each original file.
        """
        for old, new in self.f_name_map.items():
            try:
                read_path = Path(os.environ["DATA_PATH"]) / old
                write_path = Path(os.environ["DATA_PATH"]) / new
                if "ZIP" in old:
                    shutil.unpack_archive(read_path, write_path, "zip")
                else:
                    shutil.unpack_archive(read_path, write_path)
                read_path.unlink()  # delete original file
            except OSError:
                print(f"Did not unpack {read_path}")

    def split_data_characters(self) -> None:
        """Split labelled character data into train, dev (validation) and test sets."""
        read_path: Path = Path(os.environ["DATA_PATH"]) / "characters"
        character_path: Path = (
            Path(os.environ["DATA_PATH"]) / "characters" / "monkbrill2"
        )
        for directory in character_path.iterdir():
            shutil.move(str(directory.resolve()), str(read_path.resolve()))
        character_path.rmdir()

        for letter_dir in read_path.iterdir():
            for subset in self.data_split:
                # Ensure the subset and letter directories exist
                (read_path / subset / letter_dir.name).mkdir(
                    parents=True, exist_ok=True
                )
            img_paths: List[Path] = list(letter_dir.iterdir())
            split_train: int = int(len(img_paths) * self.data_split["train"])
            split_dev: int = split_train + int(len(img_paths) * self.data_split["dev"])

            split_indices: Dict[str, List[Path]] = {
                "train": img_paths[:split_train],
                "dev": img_paths[split_train:split_dev],
                "test": img_paths[split_dev:],
            }

            for subset, img_paths in split_indices.items():
                for img_path in img_paths:
                    shutil.move(
                        img_path,
                        read_path / subset / letter_dir.name / img_path.name,
                    )

            letter_dir.rmdir()

    def assert_data_characters_correct(self) -> bool:
        """Assert that the labelled character data exists and has the correct format."""
        read_path = Path(os.environ["DATA_PATH"]) / "characters"
        if not read_path.exists():
            return False
        if set([x.name for x in read_path.iterdir()]) != {"train", "dev", "test"}:
            return False
        for subset in read_path.iterdir():
            if len(list(subset.iterdir())) != self.directory_size_characters:
                return False
        return True

    def assert_train_augmented(self) -> bool:
        """Assert that the train data is augmented."""
        dalet = Path(os.environ["DATA_PATH"]) / "characters" / "train" / "Dalet"
        truth_value = False
        try:
            if len(list(dalet.iterdir())) != 72:  # downloaded number of chars
                truth_value = True
        except FileNotFoundError:
            pass  # this is ok because we handle the truth_value
        return truth_value

    def assert_data_correct(self) -> bool:
        """Assert that all data exists and is in the correct format."""
        corr_char = self.assert_data_characters_correct()
        print("Character data correct?", corr_char)
        corr_font = FontImages().assert_data_correct()
        print("Font data correct?", corr_font)
        corr_frag = self.assert_data_fragments_correct()
        print("Fragment data correct?", corr_frag)
        corr_train_aug = self.assert_train_augmented()
        print("Train data augmented?", corr_train_aug)
        truth_agree = corr_char and corr_font and corr_frag
        return True if truth_agree else False

    def split_data_fragments(self) -> None:
        """Split DSS fragments into train, dev (validation) and test sets."""
        read_path: Path = Path(os.environ["DATA_PATH"]) / "fragments"
        try:
            shutil.rmtree(read_path / "__MACOSX")
        except FileNotFoundError:
            print('Folder "__MACOSX" already removed.')

        for subset in self.data_split:  # make train, dev, test
            (read_path / subset).mkdir()

        # delete non-binarized images
        frag_paths: list = list((read_path / "image-data").iterdir())
        frags_binarized: list = [fp for fp in frag_paths if "binarized" in fp.name]
        frags_delete: set = set(frag_paths).difference(set(frags_binarized))
        for frag in frags_delete:
            frag.unlink()
        frag_paths = frags_binarized

        split_train: int = int(len(frag_paths) * self.data_split["train"])
        split_dev: int = split_train + int(len(frag_paths) * self.data_split["dev"])

        split_indices: list = [
            frag_paths[:split_train],
            frag_paths[split_train:split_dev],
            frag_paths[split_dev:],
        ]
        for idx, frag_paths in enumerate(split_indices):
            for frag in frag_paths:
                shutil.move(
                    frag, read_path / list(self.data_split.keys())[idx] / frag.name
                )

        (read_path / "image-data").rmdir()  # delete empty folder

    def assert_data_fragments_correct(self) -> bool:
        """Assert that the fragment data exists and is in the correct format."""
        read_path = Path(os.environ["DATA_PATH"]) / "fragments"
        if not read_path.exists():
            return False
        if set([x.name for x in read_path.iterdir()]) != {"train", "dev", "test"}:
            return False
        for subset in read_path.iterdir():
            # a file with "binarized" in the name should be in each subset
            if "binarized" not in list(subset.iterdir())[0].name:
                return False
        return True

    def create_font_data(self):
        """Create font data.

        Expects data/font_characters/Habbakuk.TTF
        """
        font_data = FontImages()
        if not font_data.assert_data_correct():
            font_data.create_images()
            font_data.augment_data()

    def augment_train_data(self):
        """Augment and balance the train data"""
        print("Augmenting train data.")
        read_path: Path = Path(os.environ["DATA_PATH"]) / "characters" / "train"
        for letter_dir in read_path.iterdir():
            original_images = list(letter_dir.iterdir())
            length = len(original_images)
            max_kernel = (240 - length) / 2 / length + 2
            if max_kernel >= 2.6:
                max_kernel = min(round(max_kernel), 5)
                for j in original_images:
                    img_path = str(j)
                    self.augmenter.dilate_image(img_path, 3, max_kernel)
                    self.augmenter.erosion_image(img_path, 3, max_kernel)
            new_len = len(list(letter_dir.iterdir()))
            try:  # to make the program runnable if you are not on linux
                if new_len < 160:
                    reps = 4 - new_len // 50
                    self.augmenter.elastic_morphs(letter_dir, reps)
            except Exception as e:
                print(e)
    
    def assert_style_data_correct(self) -> bool:
        """Assert that the style data exists."""
        style_chars = Path(os.environ["DATA_PATH"]) / "character_style"
        style_frags = Path(os.environ["DATA_PATH"]) / "fragment_styles"
        if style_chars.exists() and style_frags.exists():
            return True
        return False


if __name__ == "__main__":
    data_build = DatasetBuilder()
    if not data_build.assert_data_correct():
        data_build.download_all_data()
        data_build.unpack_rename_data()
        data_build.split_data_characters()
        data_build.split_data_fragments()
        data_build.create_font_data()
    if not data_build.assert_train_augmented():
        data_build.augment_train_data()
    if not data_build.assert_style_data_correct():
        data_build.download_style_data()
