import pandas as pd
import shutil
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    available_files = ["dev", "train", "test"]
    if str(args.dataset) in available_files:
        file_name = args.dataset
        file = os.path.join("data", file_name + ".tsv")
        df = pd.read_csv(file, sep="\t")
        col = "accent"
        # Rows with accent labels
        accented_df = df[(df[col].notnull()) & (df[col] != "NAN")]
        label_list = accented_df[col].unique()
        label_list.sort()
        label_dict = dict(zip(label_list, [_ for _ in range(len(label_list))]))
        print(label_dict)
        accented_df = accented_df.reset_index(drop=True)
        accented_df["label"] = accented_df[col].apply(lambda x: label_dict[x])
        # Save data dictionary with accent labels
        accented_df.to_csv(os.path.join("data", file_name + "_filtered.tsv"), sep="\t")
        # Move files to data/clips
        dst_path = os.path.join("data", "clips")
        src_path = os.path.join(
            "cv-corpus-7.0-2021-07-21-en.tar.gz",
            "cv-corpus-7.0-2021-07-21",
            "en",
            "clips",
        )
        file_list = list(accented_df["path"])
        for file in file_list:
            src = os.path.join(src_path, file)
            dst = os.path.join(dst_path, file)
            shutil.copyfile(src, dst)
    else:
        print("Wrong argument.\n Usage: python preprocess dev")