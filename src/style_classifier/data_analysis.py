import sys
import pandas as pd
from pathlib import Path

file = Path(__file__).resolve()
parent, project_root = file.parent, file.parents[2]
sys.path.append(str(project_root))

from src.data_handler.hebrew import HebrewStyles

file = Path(__file__).resolve()
parent, project_root = file.parent, file.parents[2]
sys.path.append(str(project_root))

df = pd.read_pickle("style_output_df.pkl")

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
