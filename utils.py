import pandas as pd
import re

def Load(seqs="train_sequences.csv", labels="train_labels.csv"):

    train_seqs = pd.read_csv(seqs)

    train_seqs = train_seqs[~train_seqs["sequence"].str.contains("X")
        & ~train_seqs["sequence"].str.contains("-")].reset_index(drop=True)
    train_seqs = train_seqs[["target_id","sequence"]]

    train_seqs = train_seqs[train_seqs["sequence"].str.len() > 20]

    train_seqs["sequence"] = train_seqs["sequence"].str.strip()
    train_seqs["sequence"] = train_seqs["sequence"].str.replace("G", "1")
    train_seqs["sequence"] = train_seqs["sequence"].str.replace("C", "2")
    train_seqs["sequence"] = train_seqs["sequence"].str.replace("A", "3")
    train_seqs["sequence"] = train_seqs["sequence"].str.replace("U", "4")

    train_seqs["sequence"] = train_seqs["sequence"].apply(lambda sequence: [int(molecule) for molecule in sequence])
    train_seqs = train_seqs.explode("sequence")

    train_seqs["resid"] = train_seqs.groupby("target_id").cumcount() + 1
    train_seqs["target_num"] = train_seqs["target_id"] + "_" + train_seqs["resid"].astype(str)

    train_labels = pd.read_csv(labels)
    train_labels["target_id"] = train_labels["ID"].apply(lambda x: re.split("_\d+", x)[0])
    train_labels = train_labels.groupby("target_id").filter(lambda x: ~(x.isna().any().any()))
    train_labels = train_labels.drop("target_id", axis=1)


    train_data = pd.merge(train_seqs, train_labels, left_on="target_num", right_on="ID", how="left").drop(["target_num","resid_x"], axis=1)

    train_data = train_data.rename({"resid_y": "resid", "sequence": "molecule"}, axis=1)

    train_data = train_data.dropna()

    train_data = train_data[train_data["resid"] <= 21]

    return train_data