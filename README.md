# Handwriting Recognition 2021

We present a handwriting recognition pipeline for character segmentation, recognition and time period classification.
Dead Sea Scroll data is used.

## Requirements

Instructions are targeted towards Debian/Ubuntu based systems.
Consult the documentation at <https://docs.python.org/3/library/venv.html> for handling virtual environments on other operating systems.
Firstly, install Python, pip and venv.

    sudo apt install python3 python3-pip python3-venv

Second, navigate to the project's root directory.

    cd ~/path/to/project

Initialize a virtual environment called env using the `venv` module.

    python3 -m venv env

Activate the virtual environment env.

    source env/bin/activate

Install the required project modules into the virtual environment.

    pip3 install -r requirements.txt

You may wish to test that the `tensorflow` in particular was correctly installed.

    python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

After using the software you may wish to deactivate the virtual environment as follows.

    deactivate

To enable elastic morphing during data augmentation run the following commands.

    cd ~/path/to/project
    cd src/data_handler/imagemorph
    python3 setup.py build

Lastly, ensure to manually add `.env` to the root directory of the project.
This will for example enable downloading the password protected data.
Contact the project authors to request the `.env` file.

## Running

Navigate to the project's root directory.

    cd ~/path/to/project

Download and build the dataset as well as train the recognizer and classifier models.

    python3 src/main.py --train 
    
Optional parameter for faster learning custom models.

    python3 src/main.py --train --train-fast
    
Test the models on given input fragments and output transcriptions and classifications

    python3 src/main.py --test path/to/input_dir path/to/output_dir
    
Training and testing can also be combined, for example:

    python3 src/main.py --train --train-fast --test data/fragments/ data/out

**Handwriting Recognition evaluation compliance:** The following will create recognition and classification output in a `results` directory.

    python3 src/main.py path/to/test/fragments
    
## Testing

Navigate to the project's root directory.

    cd ~/path/to/project

Run all tests using the `unittest` module.

    python3 -m unittest discover

## Architecture

Go to [ARCHITECTURE.md](ARCHITECTURE.md) to learn more about this project's architecture.

## Contributing

Go to [CONTRIBUTING.md](CONTRIBUTING.md) to learn more about contributing to this repository.

## Data

Go to [DATA.md](data/DATA.md) to learn more about loading data into this repository; this will be necessary to run the pipeline.
