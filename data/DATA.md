# Dead Sea Scrolls Training Data

The pipeline automatically downloads all training data.
Ensure that the `.env` file is present in the project's root directory to make downloading from password protected sources possible.
We use a train-dev-test split of 80%-10%-10% for all data.
An explanation of the train-dev-test split can be found on <https://cs230.stanford.edu/blog/split/>.
The data loader expects the following structure of the data.

    data/
        characters/
            train/
                Alef/
                    *.pgm 
                .../
                    *.pgm   # 300*0.8  images for each Hebrew character in pgm format
                Zayin/
                    *.pgm
            dev/
                Alef/
                    *.pgm 
                .../
                    *.pgm   # 300*0.1 images for each Hebrew character in pgm format
                Zayin/
                    *.pgm
            test/
                Alef/
                    *.pgm 
                .../
                    *.pgm   # 300*0.1 images for each Hebrew character in pgm format
                Zayin/
                    *.pgm

        fragments/
            train/
                ...
            dev/
                ...
            test/
                ...

        character_styles/
            ...

        fragment_styles/
            ...

## Optional/Deprecated: Data download with Selenium

You may wish to use `selenium` to download all data.
First install the package.

    pip3 install selenium

Then download the `geckodriver` for Firefox from <https://github.com/mozilla/geckodriver/releases>.
Choose the appropriate download for your operating system.
Then unpack the file.
Lastly add the driver to your PATH.

    # example for linux
    cd ~/Downloads
    wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz
    tar -xvf geckodriver-v0.29.1-linux64.tar.gz
    mv  geckodriver /usr/local/bin

Further instructions can be found (here)[https://selenium-python.readthedocs.io/installation.html].