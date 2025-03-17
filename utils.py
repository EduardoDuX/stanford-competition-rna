import pandas as pd

def LoadAndTransform(seqs="train_sequences.csv", labels="train_labels.csv"):
    train_seqs = pd.read_csv(seqs)

    train_seqs_new = train_seqs.copy()
    train_seqs_new["resname"] = train_seqs_new["sequence"].apply(lambda x: list(x))
    train_seqs_new = train_seqs_new[["target_id", "sequence", "resname"]].explode("resname", ignore_index=True)
    train_seqs_new["resid"] = train_seqs_new.groupby("target_id").cumcount() + 1
    train_seqs_new["target_id"] = train_seqs_new["target_id"] + "_" + train_seqs_new["resid"].astype(str)

    train_labels = pd.read_csv(labels)

    train_data = pd.merge(train_seqs_new[["target_id", "sequence"]], train_labels, left_on="target_id", right_on="ID").drop("target_id", axis=1)

    return train_data