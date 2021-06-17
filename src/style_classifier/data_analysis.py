import sys
import pandas as pd
from pathlib import Path
from tap import Tap

sys.path.append(str(Path(__file__).parents[2].resolve()))

from src.data_handler.hebrew import HebrewStyles


class ArgParser(Tap):
    """Argument parser for the style classifier analysis."""

    df_path: Path  # dataframe created after training the classifier

    def configure(self) -> None:
        """Configure the argument parser."""
        self.add_argument("df_path")


def main() -> None:
    """Analyze the style classifier output."""
    ap = ArgParser()
    args = ap.parse_args()

    df = pd.read_pickle(args.df_path)

    df["style"] = df["true"].map(lambda el: HebrewStyles.style_li[el])

    df["correct"] = df["pred"] == df["true"]
    groupby_char = df.groupby(df["char"])["correct"]
    acc = (groupby_char.sum() / groupby_char.size()).to_frame(name="acc")
    acc["occurrences"] = groupby_char.size()
    for style in HebrewStyles.style_li:
        style_df = df.loc[
            df["style"] == style,
        ]
        acc[style.lower()] = style_df.groupby(["char"]).size()
    print(acc.sort_values(by="acc", ascending=False))


if __name__ == "__main__":
    main()
