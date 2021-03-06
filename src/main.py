import argparse
from pathlib import Path
import os
import sys
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(str(Path(__file__).parents[1].resolve()))

from src.data_handler.dataset_builder import DatasetBuilder
from src.recognizer.train_test import TrainTest
from src.recognizer.transcriber import Transcriber
from src.segmentor.character_segmentation import WriteParams, segment_characters
from src.style_classifier.style_train import train_style_classifier
from src.style_classifier.style_predict import date_fragments

parser = argparse.ArgumentParser(description="Control the pipeline.")
parser.add_argument(
    "input_dir",
    type=Path,
    nargs='?',
    help="HWR course compliance: \
        provide an input directory with one or more fragment images"
)
parser.add_argument(
    "--train",
    action="store_true",
    help="run train pipeline for character recognizer and style classifier",
)
parser.add_argument(
    "--test",
    nargs=2,
    type=Path,
    help="provide first an input and second an output directory. \
        Test character recognizer and style classifier",
)
parser.add_argument(
    "--train-fast",
    action="store_true",
    help="train faster by using a simpler architecture. \
        Results in less recognition and classification accuracy.",
)


def get_work_dir() -> Path:
    """Create a cleared working directory for the pipeline's intermediate storage."""
    # make directory
    work_dir = Path(os.environ["DATA_PATH"]) / "work_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    # delete files and folders in directory
    for filename in os.listdir(work_dir):
        file_path = Path(work_dir / filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    return work_dir


def pipeline_train(train_fast) -> None:
    """
    Download and preprocess all data.
    Train the character recognizer on all available data.
    Train the style classifier on all available data.
    """
    data_builder = DatasetBuilder()
    data_builder.build_data_set()
    char_rcg = TrainTest()
    model_arch = "custom" if train_fast else "dense_net_121"
    char_rcg.train_full_model(model_arch=model_arch)
    train_style_classifier(model_arch=model_arch)


def pipeline_test(input_dir: Path, output_dir: Path) -> None:
    """
    Transcribe fragments and date them by style of an epoch.

    input_dir: Read one or more fragments from this directory.
    output_dir: Write one or more transcribed fragments to this directory.
    """
    work_dir = get_work_dir()
    segment_characters(input_dir, work_dir, WriteParams())
    transcriber = Transcriber()
    transcriber.transcribe_fragments(input_dir=work_dir, output_dir=output_dir)
    date_fragments(input_dir=work_dir, output_dir=output_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.input_dir:
        # compliance with HWR course
        pipeline_test(input_dir=args.input_dir, output_dir=Path("results"))
    elif args.train and args.test:
        pipeline_train(args.train_fast)
        pipeline_test(input_dir=args.test[0], output_dir=args.test[1])
    elif args.train:
        pipeline_train(args.train_fast)
    elif args.test:
        pipeline_test(input_dir=args.test[0], output_dir=args.test[1])
    else:
        print("Provide arguments to the pipeline! Use -h for help.")
