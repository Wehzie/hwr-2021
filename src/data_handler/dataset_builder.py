import os
import re
import requests
import shutil
from pathlib import Path

from dotenv import load_dotenv

class DatasetBuilder():

    load_dotenv()   # load environment variables from .env file
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
        "train_dev": 0.1,
        "train_test": 0.1,
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
            for chunk in r.iter_content(chunk_size=128):
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

    def split_data(self):
        '''
        Split data into train, dev (validation) and test sets.
        '''
        NotImplemented
        

    def assert_data_correct(self):
        '''
        Assert that the data exists and is in the correct format.
        NOTE this may be true in different formats
        '''
        NotImplemented


def download_data_selenium():
    '''
    Download data using Selenium.
    '''
    from selenium import webdriver
    import json

    # filename: monkbrill.tar.gz
    # description: labeled hebrew characters
    url = ''

    # initialize webdriver
    driver = webdriver.Firefox()
    driver.get(url)

    # enter password onto website
    pass_field = driver.find_element_by_id('password')
    pass_field.send_keys('')

    # submit password form
    submit_button = driver.find_element_by_id('password-submit')
    submit_button.click()

    headers = {
    "User-Agent":
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"
    }
    s = requests.session()
    s.headers.update(headers)

    for cookie in driver.get_cookies():
        c = {cookie['name']: cookie['value']}
        s.cookies.update(c)

    with open('safe_cookie', 'w+') as f:
        json.dump(driver.get_cookies(), f, indent=True)
    
    # download data
    url = ''
    r = s.get(url, stream=True)
    with open('out.tar.gz', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

def download_data_cookie():
    '''
    Download data using requests with cookies provided by Selenium.
    Does not require that Selenium is installed.
    Requires that a fresh cookie can be loaded.
    '''
    import json
    headers = {
    "User-Agent":
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"
    }

    s = requests.Session()
    s.headers.update(headers)

    cookies = None
    with open('safe_cookie', 'r') as f:
        cookies = json.load(f)

    for cookie in cookies:
        print(cookie)
        c = {cookie['name']: cookie['value']}
        s.cookies.update(c)
    
    # download data
    url = ''
    r = s.get(url, stream=True)
    with open('out.tar.gz', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


if __name__ == "__main__":
    dataset_builder = DatasetBuilder()